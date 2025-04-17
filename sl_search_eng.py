import streamlit as st
import datetime
import json

# --- Mock Search Function ---


def milvus_search_function(query):
    # Simulate returning ranked video paths
    return ["test_1k_compress/video7020.mp4",
            "test_1k_compress/video7021.mp4",
            "test_1k_compress/video7021.mp4",
            "test_1k_compress/video7021.mp4",
            "test_1k_compress/video7021.mp4",
            "test_1k_compress/video7021.mp4",]


def get_video_id(video_path):
    # Extract video ID from the path
    return video_path.split('/')[-1].split('.')[0]


def get_video_ids(video_paths):
    # Extract video IDs from the paths
    return [get_video_id(path) for path in video_paths]


def log_video_click(video_path, query, ranking):
    log_entry = {
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "query": query,
        "clicked_video": get_video_id(video_path),
        "ranking": ranking,
    }
    with open("query_click_logs.json", "a") as f:
        f.write(json.dumps(log_entry) + "\n")
    st.toast(f"Logged click: {video_path}", icon="✅")


st.markdown("#### IS 4200 Final Project")
st.title("Semantic Video Search Engine")

# Query input
query = st.text_input("Enter your search query:")
if query:
    results = milvus_search_function(query)
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
