import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.express as px

st.set_page_config(page_title="流程评分Demo", layout="wide")
st.title("📊 自动流程评分原型系统")

# 上传CSV文件
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

if df is None:
    st.subheader("🧪 未上传数据，使用内置模拟数据")
    def generate_process_data():
        np.random.seed(42)
        process_ids = [f"P{str(i).zfill(3)}" for i in range(1, 6)]
        events_per_process = [3, 4, 5, 3, 4]
        org_units = ['BranchA', 'BranchB', 'HQ']
        activities = ['Receive Request', 'Review', 'Approve', 'Finalize', 'Archive']
        statuses = ['success', 'failed']

        rows = []
        start_base = datetime(2025, 8, 1, 9, 0, 0)

        for pid, num_events in zip(process_ids, events_per_process):
            current_time = start_base + timedelta(minutes=np.random.randint(0, 30))
            for eid in range(num_events):
                event_id = f"E{eid+1:02d}"
                activity = activities[eid % len(activities)]
                duration = timedelta(minutes=np.random.randint(5, 30))
                start_time = current_time
                end_time = start_time + duration
                performer = f"User{np.random.choice(list('ABCDE'))}"
                org_unit = np.random.choice(org_units)
                status = np.random.choice(statuses if eid == num_events - 1 else ['success'])
                rows.append([pid, event_id, activity, start_time, end_time, performer, org_unit, status])
                current_time = end_time + timedelta(minutes=np.random.randint(1, 10))

        df = pd.DataFrame(rows, columns=[
            'process_id', 'event_id', 'activity_name', 'start_time', 'end_time', 'performer', 'org_unit', 'status'
        ])
        return df

    df = generate_process_data()

# 计算流程指标
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

# 评分逻辑
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

st.subheader("🔍 原始流程日志数据")
st.dataframe(df, use_container_width=True)

metrics_df = compute_metrics(df)
scored_df = score_processes(metrics_df)

st.subheader("📈 流程指标 + 自动评分")
st.dataframe(scored_df, use_container_width=True)

st.subheader("📊 流程评分雷达图")
fig = px.line_polar(scored_df, r='score', theta='process_id', line_close=True,
                    title="流程评分雷达图", markers=True)
st.plotly_chart(fig, use_container_width=True)

st.subheader("📊 流程指标柱状图")
fig_bar = px.bar(scored_df.sort_values("score", ascending=False), x='process_id', y='score',
                 title="流程评分柱状图", text='score')
st.plotly_chart(fig_bar, use_container_width=True)

st.markdown("---")
st.markdown("🔁 当前支持上传CSV文件进行评分，也可以使用内置模拟数据。建议字段包括：流程ID、事件ID、事件名称、事件开始/结束时间、执行人、执行机构、状态等")
