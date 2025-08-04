import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.express as px

# --- 页面基础配置 ---
st.set_page_config(page_title="流程评分Demo", layout="wide")
primary_color = '#636efa'    # 主色 蓝
secondary_color = '#ab63fa'  # 辅色 紫
background_color = '#f5f7fa' # 页面背景
font_family = 'Arial, sans-serif'

# 自定义CSS美化
st.markdown(f"""
    <style>
    .reportview-container {{
        background-color: {background_color};
        font-family: {font_family};
        color: #222;
    }}
    .css-1d391kg {{
        padding-top: 1rem;
        padding-bottom: 2rem;
    }}
    .css-18e3th9 {{
        padding-left: 3rem;
        padding-right: 3rem;
    }}
    .stButton>button {{
        background-color: {primary_color};
        color: white;
        font-weight: 600;
        border-radius: 8px;
        padding: 8px 24px;
        border: none;
        transition: background-color 0.3s ease;
    }}
    .stButton>button:hover {{
        background-color: {secondary_color};
        color: white;
    }}
    .css-1hynsf2 p {{
        font-size: 18px;
    }}
    </style>
""", unsafe_allow_html=True)

st.title("📊 自动流程评分原型系统")
st.write("上传流程日志数据或使用模拟数据，自动计算流程绩效评分，并生成可视化报告。")

# --- 上传区 ---
st.subheader("📤 上传CSV文件进行评分")
uploaded_file = st.file_uploader("上传包含流程日志的CSV文件", type="csv")

default_fields = {
    'process_id': '流程ID',
    'event_id': '事件ID',
    'activity_name': '事件名称',
    'start_time': '事件开始时间',
    'end_time': '事件结束时间',
    'performer': '事件执行人',
    'org_unit': '事件执行机构',
    'status': '流程状态'
}

field_mapping = {}

df = None
if uploaded_file is not None:
    raw_df = pd.read_csv(uploaded_file)
    st.success("✅ 文件上传成功，检测到以下字段：")
    st.write(list(raw_df.columns))

    st.subheader("🛠️ 字段映射配置")
    for key, label in default_fields.items():
        options = [col for col in raw_df.columns]
        selected = st.selectbox(f"选择对应的【{label}】字段:", options, key=key)
        field_mapping[key] = selected

    try:
        df = raw_df.rename(columns=field_mapping)
        df['start_time'] = pd.to_datetime(df['start_time'])
        df['end_time'] = pd.to_datetime(df['end_time'])
        st.success("✅ 字段映射成功，已转换为内部格式")
    except Exception as e:
        st.error(f"❌ 字段转换出错: {e}")
        st.stop()
else:
    st.info("📥 请上传包含流程日志的CSV文件...")

# --- 模拟数据生成 ---
if df is None:
    st.subheader("🧪 未上传数据，使用内置模拟数据")

    def generate_process_data():
        np.random.seed(42)
        process_ids = [f"P{str(i).zfill(4)}" for i in range(1, 51)]
        events_per_process = np.random.randint(5, 15, size=len(process_ids))
        org_units = ['BranchA', 'BranchB', 'HQ', 'BranchC', 'BranchD']
        activities = ['Receive Request', 'Review', 'Approve', 'Finalize', 'Archive', 'Validate', 'Escalate', 'Notify']
        statuses = ['success', 'failed']

        rows = []
        start_base = datetime(2025, 8, 1, 9, 0, 0)

        for pid, num_events in zip(process_ids, events_per_process):
            current_time = start_base + timedelta(minutes=np.random.randint(0, 1440))
            for eid in range(num_events):
                event_id = f"E{eid+1:03d}"
                activity = np.random.choice(activities)
                duration = timedelta(minutes=np.random.randint(5, 60))
                start_time = current_time
                end_time = start_time + duration
                performer = f"User{np.random.choice(list('ABCDEFGHIJ'))}"
                org_unit = np.random.choice(org_units)
                if eid == num_events - 1:
                    status = np.random.choice(statuses, p=[0.9, 0.1])  # 最后一步90%成功，10%失败
                else:
                    status = 'success'
                rows.append([pid, event_id, activity, start_time, end_time, performer, org_unit, status])
                current_time = end_time + timedelta(minutes=np.random.randint(1, 20))

        df = pd.DataFrame(rows, columns=[
            'process_id', 'event_id', 'activity_name', 'start_time', 'end_time', 'performer', 'org_unit', 'status'
        ])
        return df

    df = generate_process_data()

# --- 计算指标 ---
def compute_metrics(df):
    process_metrics = []

    for pid, group in df.groupby("process_id"):
        group_sorted = group.sort_values("start_time")
        total_duration = (group_sorted["end_time"].max() - group_sorted["start_time"].min()).total_seconds() / 60.0
        avg_activity_duration = (group_sorted["end_time"] - group_sorted["start_time"]).dt.total_seconds().mean() / 60.0
        num_activities = group_sorted.shape[0]
        org_changes = group_sorted["org_unit"].nunique()
        failed_steps = group_sorted[group_sorted["status"] == "failed"].shape[0]
        wait_times = (group_sorted["start_time"].iloc[1:].reset_index(drop=True) - 
                      group_sorted["end_time"].iloc[:-1].reset_index(drop=True)).dt.total_seconds() / 60.0
        avg_wait_time = wait_times.mean() if not wait_times.empty else 0.0

        process_metrics.append({
            "process_id": pid,
            "total_duration_min": total_duration,
            "avg_activity_duration_min": avg_activity_duration,
            "num_activities": num_activities,
            "num_org_units": org_changes,
            "num_failed_steps": failed_steps,
            "avg_wait_time_min": avg_wait_time
        })

    return pd.DataFrame(process_metrics)

# --- 评分函数 ---
def score_processes(metrics_df):
    norm_df = (metrics_df.drop(columns=['process_id']) - metrics_df.drop(columns=['process_id']).min()) / \
              (metrics_df.drop(columns=['process_id']).max() - metrics_df.drop(columns=['process_id']).min())
    norm_df.fillna(0, inplace=True)

    weights = {
        'total_duration_min': 0.35,
        'avg_activity_duration_min': 0.20,
        'num_activities': 0.10,
        'num_org_units': 0.15,
        'num_failed_steps': 0.10,
        'avg_wait_time_min': 0.10
    }

    score = 100 - (norm_df * pd.Series(weights)).sum(axis=1) * 100
    metrics_df["score"] = score.round(1)
    return metrics_df

# --- 显示原始数据 ---
st.subheader("🔍 原始流程日志数据")
st.dataframe(df, use_container_width=True)

# --- 计算并评分 ---
metrics_df = compute_metrics(df)
scored_df = score_processes(metrics_df)

# --- 图表美化参数 ---
import plotly.express as px

def style_fig(fig):
    fig.update_layout(
        font=dict(family=font_family, size=14, color='#222'),
        paper_bgcolor=background_color,
        plot_bgcolor='white',
        margin=dict(l=40, r=40, t=50, b=40),
        title_font=dict(size=20, family=font_family),
        xaxis=dict(showgrid=True, gridcolor='lightgray', zeroline=False),
        yaxis=dict(showgrid=True, gridcolor='lightgray', zeroline=False)
    )
    return fig

# --- 流程评分雷达图 ---
st.subheader("📊 流程评分雷达图")
fig_radar = px.line_polar(
    scored_df,
    r='score',
    theta='process_id',
    line_close=True,
    markers=True,
    title="流程评分雷达图",
    color_discrete_sequence=[primary_color]
)
fig_radar.update_traces(fill='toself', fillcolor='rgba(99,110,250,0.2)')
fig_radar.update_layout(
    polar=dict(
        bgcolor='white',
        radialaxis=dict(showline=True, linewidth=1, gridcolor='lightgray', gridwidth=0.5, tickfont=dict(size=10)),
        angularaxis=dict(tickfont=dict(size=10))
    )
)
st.plotly_chart(fig_radar, use_container_width=True)

st.markdown("---")

# --- 报告部分 ---
st.header("📋 流程评分报告")

col1, col2, col3 = st.columns(3)
col1.metric("平均总耗时（分钟）", f"{scored_df['total_duration_min'].mean():.1f}")
col2.metric("平均失败步骤数", f"{scored_df['num_failed_steps'].mean():.1f}")
col3.metric("平均流程评分", f"{scored_df['score'].mean():.1f}")

st.subheader("🚀 评分最高Top 10流程（表现最好）")
top10 = scored_df.sort_values('score', ascending=False).head(10)
st.dataframe(top10[['process_id', 'score', 'total_duration_min', 'num_failed_steps']], use_container_width=True)

st.subheader("🐢 评分最低Bottom 10流程（表现最差）")
bottom10 = scored_df.sort_values('score').head(10)
st.dataframe(bottom10[['process_id', 'score', 'total_duration_min', 'num_failed_steps']], use_container_width=True)

st.subheader("⏱️ 流程总耗时分布")
fig_hist = px.histogram(
    scored_df,
    x='total_duration_min',
    nbins=30,
    labels={'total_duration_min': '总耗时（分钟）'},
    title="流程总耗时分布直方图",
    color_discrete_sequence=[primary_color]
)
fig_hist = style_fig(fig_hist)
st.plotly_chart(fig_hist, use_container_width=True)

st.subheader("⚠️ 失败步骤数分布")
fig_box = px.box(
    scored_df,
    y='num_failed_steps',
    points='all',
    title="流程失败步骤数分布箱型图",
    color_discrete_sequence=[secondary_color]
)
fig_box.update_layout(
    yaxis=dict(showgrid=True, gridcolor='lightgray', zeroline=False)
)
fig_box = style_fig(fig_box)
st.plotly_chart(fig_box, use_container_width=True)

st.subheader("📊 失败步骤比例统计")
total_failed_steps = scored_df['num_failed_steps'].sum()
total_steps = scored_df['num_activities'].sum()
failed_rate = total_failed_steps / total_steps if total_steps > 0 else 0
st.write(f"总失败步骤占所有步骤的比例约为：**{failed_rate:.2%}**")

st.markdown("---")
st.markdown("🔁 支持上传CSV文件进行评分，也可使用内置模拟数据。字段映射可适配任意列名。")
