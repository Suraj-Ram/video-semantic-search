import datetime
import json
import os

import streamlit as st
from towhee import ops, pipe
from towhee.operator import PyOperator
from tqdm import tqdm

VIDEO_RET_COLLECTION = "msrvtt_vid_ret_1"
TOP_K = 10
VIDEOS_FOLDER = "./test_1k_compress"


search_pipeline = (
    pipe.input('sentence')
    .map('sentence', 'vec', ops.video_text_embedding.clip4clip(model_name='clip_vit_b32', modality='text', device='mps'))
    .map('vec', 'rows', ops.ann_search.milvus_client(collection_name=VIDEO_RET_COLLECTION, limit=TOP_K))
    .map('rows', 'videos_path', lambda rows: (os.path.join(VIDEOS_FOLDER, 'video' + str(r[0]) + '.mp4') for r in rows))
    .output('videos_path')
)


def query_videos(text):
    search_results = search_pipeline(text).to_list()
    return search_results[0][0]


def get_video_id(video_path):
    return video_path.split('/')[-1].split('.')[0]


def get_video_ids(video_paths):
    return [get_video_id(path) for path in video_paths]


def append_log_entry(log_entry):
    with open("query_click_logs.json", "a") as f:
        f.write(json.dumps(log_entry) + "\n")


def log_video_click(video_path, query, ranking):
    log_entry = {
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "query": query,
        "clicked_video": get_video_id(video_path),
        "ranking": ranking,
    }
    append_log_entry(log_entry)
    st.toast(f"Logged click: {video_path}", icon="✅")


st.markdown("#### IS 4200 Final Project")
st.title("Semantic Video Search Engine")

# Query input
query = st.text_input("Enter your search query:")

if query:
    results = query_videos(query)
    st.subheader("Top Results")

    # Display video results
    for i, video_path in enumerate(results, start=1):
        col1, col2 = st.columns([4, 5])  # Adjusted column proportions
        with col1:
            st.video(video_path, format="video/mp4", start_time=0)
        with col2:
            st.write(f"**Result {i}** - {get_video_id(video_path)}.mp4")
            if st.button(f"Watched", key=f"watch_{i}"):
                log_video_click(video_path, query, get_video_ids(results))
