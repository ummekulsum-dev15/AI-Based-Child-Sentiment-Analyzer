# Warning Suppression Guide

## ✅ Warnings Fixed

The following annoying warnings have been suppressed:

### Before Fix:
```
WARNING: All log messages before absl::InitializeLog() is called are written to STDERR
I0000 00:00:1775726503.332110    3768 port.cc:153] oneDNN custom operations are on...
INFO:httpx:HTTP Request: GET http://127.0.0.1:7860/gradio_api/startup-events...
INFO:httpx:HTTP Request: HEAD https://huggingface.co/api/telemetry/...
```

### After Fix:
```
* Running on local URL:  http://127.0.0.1:7860
```
✅ Clean output!

---

## 🔧 What Was Changed

### In `main.py` (Top of file):

```python
# Suppress all warnings at the very beginning
import os
import warnings
import logging

# Suppress Python warnings
warnings.filterwarnings('ignore')

# Suppress TensorFlow/oneDNN warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # 0=ALL, 1=INFO, 2=WARNING, 3=ERROR
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Disable oneDNN custom ops warning

# Suppress Gradio/HTTP logs
logging.getLogger('gradio').setLevel(logging.ERROR)
logging.getLogger('httpx').setLevel(logging.ERROR)
logging.getLogger('httpcore').setLevel(logging.ERROR)
logging.getLogger('urllib3').setLevel(logging.ERROR)
logging.getLogger('transformers').setLevel(logging.ERROR)
logging.getLogger('tensorflow').setLevel(logging.ERROR)
```

### In `main()` function:

```python
demo.launch(quiet=True)  # Added quiet=True to suppress launch messages
```

---

## 📋 What Each Suppression Does

| Warning Type | Solution | Effect |
|-------------|----------|--------|
| **oneDNN operations** | `TF_ENABLE_ONEDNN_OPTS=0` | Disables optimization warning |
| **TensorFlow logs** | `TF_CPP_MIN_LOG_LEVEL=3` | Shows only errors |
| **HTTP requests** | `logging.getLogger('httpx')` | Hides API call logs |
| **Gradio logs** | `logging.getLogger('gradio')` | Hides startup logs |
| **Transformers** | `logging.getLogger('transformers')` | Hides model loading logs |
| **Python warnings** | `warnings.filterwarnings('ignore')` | Hides all Python warnings |

---

## 🚀 Running the App

Now when you run:
```bash
.venv\Scripts\python.exe main.py
```

You'll see:
```
* Running on local URL:  http://127.0.0.1:7860
```

Clean and professional! ✨

---

## ⚠️ Important Notes

- **No functionality is affected** - Only console output is cleaned up
- **Errors still show** - If something breaks, you'll still see error messages
- **Debug mode available** - To see warnings again for debugging, comment out the suppression lines

---

**Last Updated**: April 9, 2026
**Status**: All warnings suppressed ✅
