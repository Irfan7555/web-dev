import streamlit as st
import json
import os
from datetime import datetime
from bokeh.models.widgets import Button
from bokeh.models import CustomJS
from bokeh.layouts import column
from streamlit_bokeh_events import streamlit_bokeh_events

st.title("🎙️ Speech-to-Text Logger")

# JSON file path
LOG_FILE = "../streamlit-webrtc/conversation_log.json"

# Ensure file exists
if not os.path.exists(LOG_FILE):
    with open(LOG_FILE, "w") as f:
        json.dump([], f)

# Speak Button
start_button = Button(label="Speak", width=100)
start_button.js_on_event("button_click", CustomJS(code="""
    window.recognition = new webkitSpeechRecognition();
    window.recognition.continuous = true;
    window.recognition.interimResults = true;

    window.recognition.onresult = function (e) {
        var value = "";
        for (var i = e.resultIndex; i < e.results.length; ++i) {
            if (e.results[i].isFinal) {
                value += e.results[i][0].transcript;
            }
        }
        if (value !== "") {
            document.dispatchEvent(new CustomEvent("GET_TEXT", { detail: value }));
        }
    }
    window.recognition.start();
"""))

# Stop Button
stop_button = Button(label="Stop", width=100)
stop_button.js_on_event("button_click", CustomJS(code="""
    if (window.recognition) {
        window.recognition.stop();
    }
"""))

# Layout for buttons
button_layout = column(start_button, stop_button)

# Show the buttons and capture speech
stt_result = streamlit_bokeh_events(
    button_layout,
    events="GET_TEXT",
    key="speech_logger",
    refresh_on_update=False,
    override_height=75,
    debounce_time=0,
)

# Handle result
if stt_result and "GET_TEXT" in stt_result:
    text = stt_result["GET_TEXT"]
    timestamp = datetime.now().isoformat()

    st.write(f"🗣️ You said: **{text}**")

    # Append to JSON log
    new_entry = {"timestamp": timestamp, "text": text}

    with open(LOG_FILE, "r+") as f:
        logs = json.load(f)
        logs.append(new_entry)
        f.seek(0)
        json.dump(logs, f, indent=4)

    st.success("✅ Text saved to conversation_log.json")

# Optionally display full conversation log
if st.checkbox("Show full conversation log"):
    with open(LOG_FILE) as f:
        logs = json.load(f)
        for entry in logs:
            st.write(f"🕒 {entry['timestamp']}")
            st.write(f"💬 {entry['text']}")
            st.markdown("---")
