import streamlit as st
import pandas as pd
import time
import plotly.figure_factory as ff
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import json
import akshare as ak
import datetime
import openpyxl
import io
import pandas_ta as ta
from plotly.subplots import make_subplots
from streamlit_autorefresh import st_autorefresh
import hashlib

def check_password():
    users_df = pd.read_excel('users.xlsx')
    users = dict(zip(users_df['users'], users_df['password']))

    if 'authenticated' not in st.session_state:
        st.session_state['authenticated'] = False

    if not st.session_state['authenticated']:
        st.title("登录")
        username = st.text_input("用户名")
        password = st.text_input("密码", type="password")
        if st.button("登录"):
            # 可选：对密码做hash校验
            # password = hashlib.sha256(password.encode()).hexdigest()
            if username in users and str(users[username]) == password:
                st.session_state['authenticated'] = True
                st.success("登录成功！")
                st.rerun()
            else:
                st.error("用户名或密码错误")
        st.stop()

check_password()

def format_numeric_df(df):
        df_fmt = df.copy()
        for col in df_fmt.columns:
            if pd.api.types.is_numeric_dtype(df_fmt[col]):
                df_fmt[col] = df_fmt[col].apply(lambda x: f"{x:,.2f}" if pd.notnull(x) else x)
        return df_fmt

def parse_chinese_number(val):
    """将带有'亿'、'万'的字符串转为float"""
    if isinstance(val, str):
        val = val.replace(',', '').strip()
        if val.endswith('亿'):
            return float(val[:-1]) * 1e8
        elif val.endswith('万'):
            return float(val[:-1]) * 1e4
        else:
            return float(val)
    return float(val)

def get_b_minute_data(stock_code):
    # 获取分时数据
    df = ak.stock_zh_b_minute(symbol=stock_code, period='1', adjust="qfq")
    return df.iloc[-1:]

def main():
   
    st.set_page_config(
        page_title="Stock Analysis Dashboard",
        layout="wide",)
    lists=pd.read_csv('代码列表.csv', dtype={'代码': str})
    st.sidebar.title("股票代码列表")
    
    # 侧边栏添加股票名称搜索
    search_name = st.sidebar.text_input("股票名称搜索")
    if search_name:
        lists = lists[lists['名称'].str.contains(search_name)]
    
    selected_code = st.sidebar.selectbox(
        "选择股票代码",
        lists['代码'].tolist(),
        format_func=lambda x: lists[lists['代码'] == x]['名称'].values[0] if x in lists['代码'].values else x
    )
    selected_code = str(selected_code).zfill(6)
    if selected_code in lists['代码'].values:
        symbols = selected_code
    else:
        st.error("未找到对应的股票代码，请检查代码列表和选择框。")
        st.stop()
    st.sidebar.write('---')
    date_range = st.sidebar.select_slider(
        '选择日期',
        options=pd.date_range(start='2014-01-01', end=datetime.datetime.today()).strftime('%Y-%m-%d').tolist(),
        value=(pd.to_datetime('2025-01-01').strftime('%Y-%m-%d'), datetime.datetime.today().strftime('%Y-%m-%d'))
    )
    start_date = pd.to_datetime(date_range[0]).strftime('%Y%m%d')
    end_date = pd.to_datetime(date_range[1]).strftime('%Y%m%d')
    st.sidebar.write('---')
    adjusts=st.sidebar.selectbox(
        "选择复权类型",
        ["不复权", "前复权", "后复权"],
        index=0,
    )
    # 自动拼接市场前缀
    if symbols.startswith('6'):
        stock_code = f"sh{symbols}"
    else:
        stock_code = f"sz{symbols}"

    now = int(time.time())
 

    st.write(f"### 选中的股票代码: {symbols} - {lists[lists['代码'] == selected_code]['名称'].values[0]}")
   


    if adjusts == "不复权":
        adjust = ''
    elif adjusts == "前复权":
        adjust = 'qfq'
    elif adjusts == "后复权":
        adjust = 'hfq'
    df = ak.stock_zh_a_hist(
        symbol=symbols,
        start_date=start_date,
        end_date=end_date,
        adjust=adjust  # 不复权
    )

    if df.empty:
        st.error("未获取到历史行情数据，请检查代码或日期区间。")
        st.stop()

    # 计算技术指标
    macd = ta.macd(df['收盘'], fast=12, slow=26, signal=9)
    df['MACD'] = macd['MACD_12_26_9']
    df['MACD_signal'] = macd['MACDs_12_26_9']
    df['MACD_hist'] = macd['MACDh_12_26_9']
    df['RSI'] = ta.rsi(df['收盘'], length=14)
    bbands = ta.bbands(df['收盘'], length=20)
    df['BB_upper'] = bbands['BBU_20_2.0']
    df['BB_middle'] = bbands['BBM_20_2.0']
    df['BB_lower'] = bbands['BBL_20_2.0']

    # 在侧边栏添加技术指标选择
    tech_options = st.sidebar.multiselect(
        "选择要显示的技术指标",
        ["MACD", "RSI", "BOLL", "MA5", "MA10", "MA20"],
        default=["MACD", "RSI", "BOLL"]
    )

    # 计算均线
    if "MA5" in tech_options:
        df['MA5'] = df['收盘'].rolling(window=5).mean()
    if "MA10" in tech_options:
        df['MA10'] = df['收盘'].rolling(window=10).mean()
    if "MA20" in tech_options:
        df['MA20'] = df['收盘'].rolling(window=20).mean()

    st.dataframe(format_numeric_df(df), use_container_width=True)
    
    if not df.empty:
        start_price = df['收盘'].iloc[0]
        end_price = df['收盘'].iloc[-1]
        pct = (end_price - start_price) / start_price * 100
        st.metric("区间涨跌幅", f"{pct:.2f}%")

    stock_fhps_detail_ths_df = ak.stock_fhps_detail_ths(symbol=symbols)
    st.write("### 分红派息详情")
    col1, col2 = st.columns(2)
    # 自动查找包含“分红总额”关键字的列名
    dividend_total_col = None
    for col in stock_fhps_detail_ths_df.columns:
        if "分红总额" in col:
            dividend_total_col = col
            break

    if dividend_total_col is None:
        st.warning("未找到包含'分红总额'的列")
        dividend_value = pd.Series(dtype=float)
        dividend_date = pd.Series(dtype=str)
    else:
        # 用自动找到的列名替换所有相关代码
        dividend_value = stock_fhps_detail_ths_df.loc[stock_fhps_detail_ths_df[dividend_total_col] != '--', dividend_total_col]
        dividend_date = stock_fhps_detail_ths_df.loc[stock_fhps_detail_ths_df[dividend_total_col] != '--', 'A股除权除息日']

    # 后续用 dividend_value 和 dividend_date 变量
    with col1:
        if not dividend_date.empty:
            st.metric("分红日期", str(dividend_date.values[-1]))
        else:
            st.metric("分红日期", "无数据")
    with col2:
        if not dividend_value.empty:
            st.metric("分红总额", dividend_value.values[-1])
        else:
            st.metric("分红总额", "无数据")
    col1,col2= st.columns([2,1])
    with col1:
        st.dataframe(format_numeric_df(stock_fhps_detail_ths_df), use_container_width=True,hide_index=True)
    with col2:
        股本=ak.stock_individual_info_em(symbol=symbols)
        股本.columns=['项目','金额']
        st.dataframe(股本)
    st.write(f"#### {lists[lists['代码'] == selected_code]['名称'].values[0]}-DCF分红预测与现值表")
    col1, col2 = st.columns([2, 1])
    with col1:
        st.sidebar.write("### DCF模型估值")
        折现率 = st.sidebar.number_input("折现率", min_value=0.01, max_value=0.2, value=0.05, step=0.01, key="discount_rate")
        增长阶段 = st.sidebar.selectbox('阶段数', [1, 2, 3], index=0, key="growth_stage")

        # 获取初始分红
        if not dividend_value.empty:
            try:
                初始分红 = parse_chinese_number(dividend_value.values[-1])
            except Exception:
                初始分红 = 0.0
        else:
            初始分红 = 0.0

        年数列表, 增长率列表 = [], []
        if 增长阶段 == 1:
            col_a, col_b = st.sidebar.columns(2)
            with col_a:
                增长率 = st.number_input("增长率", min_value=0.01, max_value=0.2, value=0.05, step=0.01, key="growth_rate")
            with col_b:
                增长期间 = st.number_input("期间（年）", min_value=1, max_value=30, value=5, step=1, key="growth_period")
            年数列表 = [增长期间]
            增长率列表 = [增长率]
        elif 增长阶段 == 2:
            col_a, col_b = st.sidebar.columns(2)
            with col_a:
                增长率1 = st.number_input("阶段1增长率", min_value=0.01, max_value=0.2, value=0.05, step=0.01, key="growth_rate1")
            with col_b:
                增长期间1 = st.number_input("阶段1期间", min_value=1, max_value=30, value=5, step=1, key="growth_period1")
            col_c, col_d = st.sidebar.columns(2)
            with col_c:
                增长率2 = st.number_input("阶段2增长率", min_value=0.01, max_value=0.2, value=0.03, step=0.01, key="growth_rate2")
            with col_d:
                增长期间2 = st.number_input("阶段2期间", min_value=1, max_value=30, value=5, step=1, key="growth_period2")
            年数列表 = [增长期间1, 增长期间2]
            增长率列表 = [增长率1, 增长率2]
        elif 增长阶段 == 3:
            col_a, col_b = st.sidebar.columns(2)
            with col_a:
                增长率1 = st.number_input("阶段1增长率", min_value=0.01, max_value=0.2, value=0.05, step=0.01, key="growth_rate1")
            with col_b:
                增长期间1 = st.number_input("阶段1期间", min_value=1, max_value=30, value=5, step=1, key="growth_period1")
            col_c, col_d = st.sidebar.columns(2)
            with col_c:
                增长率2 = st.number_input("阶段2增长率", min_value=0.01, max_value=0.2, value=0.03, step=0.01, key="growth_rate2")
            with col_d:
                增长期间2 = st.number_input("阶段2期间", min_value=1, max_value=30, value=5, step=1, key="growth_period2")
            col_e, col_f = st.sidebar.columns(2)
            with col_e:
                增长率3 = st.number_input("阶段3增长率", min_value=0.01, max_value=0.2, value=0.02, step=0.01, key="growth_rate3")
            with col_f:
                增长期间3 = st.number_input("阶段3期间", min_value=1, max_value=30, value=5, step=1, key="growth_period3")
            年数列表 = [增长期间1, 增长期间2, 增长期间3]
            增长率列表 = [增长率1, 增长率2, 增长率3]

        # 生成DCF现金流表
        dcf_rows = []
        分红 = 初始分红
        年份 = []
        for idx, (years, rate) in enumerate(zip(年数列表, 增长率列表)):
            for y in range(1, int(years)+1):
                年份.append(len(年份)+1)
                分红 = 分红 * (1 + rate) if len(dcf_rows) > 0 else 分红 * (1 + rate)
                现值 = 分红 / ((1 + 折现率) ** len(年份))
                dcf_rows.append({
                    "年份": len(年份),
                    "分红预测": 分红,
                    "增长率": rate,
                    "折现率": 折现率,
                    "现值": 现值
                })

        dcf_df = pd.DataFrame(dcf_rows)
        总现值 = dcf_df["现值"].sum() if not dcf_df.empty else 0
        st.write()
        
        st.dataframe(format_numeric_df(dcf_df), use_container_width=True, hide_index=True)
        with col2:
            st.metric("DCF估算总现值", f"{总现值:,.2f}")
            st.metric('DCF每股价值', f"{总现值 / 股本.iloc.loc['总股本','金额']:,.2f}" if not 股本.empty else "无数据")
            st.metric("当前股价", f"{df['收盘'].iloc[-1]:,.2f}")

    stock_financial_analysis_indicator_df = ak.stock_financial_analysis_indicator(symbol=symbols, start_year="2020")
    st.write("### 财务分析指标")
    st.dataframe(format_numeric_df(stock_financial_analysis_indicator_df), use_container_width=True,hide_index=True)
    # 侧边栏选择要显示的曲线
  
    # 创建上下两行子图，K线在上，MACD在下
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        row_heights=[1, 1],  # 两个图都更高
        subplot_titles=("K线图与技术指标", "MACD")
    )

    # K线图
    fig.add_trace(
        go.Candlestick(
            x=df['日期'],
            open=df['开盘'],
            high=df['最高'],
            low=df['最低'],
            close=df['收盘'],
            name='K线'
        ),
        row=1, col=1
    )

    # 叠加均线和BOLL
    if "MA5" in tech_options:
        fig.add_trace(go.Scatter(x=df['日期'], y=df['MA5'], mode='lines', name='MA5'), row=1, col=1)
    if "MA10" in tech_options:
        fig.add_trace(go.Scatter(x=df['日期'], y=df['MA10'], mode='lines', name='MA10'), row=1, col=1)
    if "MA20" in tech_options:
        fig.add_trace(go.Scatter(x=df['日期'], y=df['MA20'], mode='lines', name='MA20'), row=1, col=1)
    if "BOLL" in tech_options:
        fig.add_trace(go.Scatter(x=df['日期'], y=df['BB_upper'], mode='lines', name='BOLL上轨', line=dict(dash='dot')), row=1, col=1)
        fig.add_trace(go.Scatter(x=df['日期'], y=df['BB_middle'], mode='lines', name='BOLL中轨', line=dict(dash='dash')), row=1, col=1)
        fig.add_trace(go.Scatter(x=df['日期'], y=df['BB_lower'], mode='lines', name='BOLL下轨', line=dict(dash='dot')), row=1, col=1)

    # MACD主线、信号线、柱状图
    fig.add_trace(go.Bar(x=df['日期'], y=df['MACD_hist'], name='MACD柱', marker_color='gray'), row=2, col=1)
    fig.add_trace(go.Scatter(x=df['日期'], y=df['MACD'], mode='lines', name='MACD', line=dict(color='blue')), row=2, col=1)
    fig.add_trace(go.Scatter(x=df['日期'], y=df['MACD_signal'], mode='lines', name='信号线', line=dict(color='red', dash='dash')), row=2, col=1)

    # 只在下方显示x轴标签
    fig.update_xaxes(showticklabels=False, row=1, col=1)
    fig.update_xaxes(showticklabels=True, row=2, col=1)

    # 设置整个图表高度（比如2000像素）
    fig.update_layout(
        height=800,
        title="K线图与MACD",
        xaxis_title=None,
        yaxis_title="价格",
        xaxis2_title="日期",
        yaxis2_title="MACD",
        template='plotly_white',
        showlegend=True,
        legend=dict(orientation="h")
    )

    # 自动设置MACD区间，留出上下边距
    macd_min = df[['MACD', 'MACD_signal', 'MACD_hist']].min().min()
    macd_max = df[['MACD', 'MACD_signal', 'MACD_hist']].max().max()
    margin = (macd_max - macd_min) * 1 if macd_max != macd_min else 1
    fig.update_yaxes(range=[macd_min - margin, macd_max + margin], row=2, col=1)

    st.plotly_chart(fig, use_container_width=True)

    rsi_fig = go.Figure()
    rsi_fig.add_trace(go.Scatter(x=df['日期'], y=df['RSI'], mode='lines', name='RSI', line=dict(color='purple')))
    rsi_fig.update_layout(title="RSI指标", xaxis_title="日期", yaxis_title="RSI", template='plotly_white', yaxis=dict(range=[0, 100]))
    st.plotly_chart(rsi_fig, use_container_width=True)
    # 添加下载 Excel 按钮
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df.to_excel(writer, index=False)
    data = output.getvalue()
    st.download_button(
        label="下载数据为 Excel",
        data=data,
        file_name=f"{symbols}_{start_date}_{end_date}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

   
    
    

    stock_financial_report_sina_df = ak.stock_financial_report_sina(stock=stock_code, symbol="资产负债表")
    stock_financial_report_sina_df = stock_financial_report_sina_df.T
    stock_financial_report_sina_df.columns = stock_financial_report_sina_df.iloc[0]
    income_statement=ak.stock_financial_report_sina(stock=stock_code, symbol="利润表")
    income_statement = income_statement.T
    income_statement.columns = income_statement.iloc[0]
    # 财务报表格式化
    st.write("### 财务报表")
    col1,col2,col3 = st.columns(3)
    with col1:
        st.write("#### 资产负债表")
        st.dataframe(format_numeric_df(stock_financial_report_sina_df), use_container_width=True)
    with col2:
        st.write("#### 利润表")
        st.dataframe(format_numeric_df(income_statement), use_container_width=True)
    with col3:
        st.write("#### 现金流量表")
        cash_flow_statement = ak.stock_financial_report_sina(stock=stock_code, symbol="现金流量表")
        cash_flow_statement = cash_flow_statement.T
        cash_flow_statement.columns = cash_flow_statement.iloc[0]
        st.dataframe(format_numeric_df(cash_flow_statement), use_container_width=True)

    

if __name__ == "__main__":
    main()

