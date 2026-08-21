"""
Run this directly with: python3 test_backend_streaming.py
Make sure your FastAPI backend is already running (uvicorn) before running this.

This hits /api/chat/stream directly over HTTP and times how each SSE "data:"
line arrives — isolating whether FastAPI is flushing progressively, without
involving the frontend at all.
"""
import time
import requests

url = "http://localhost:8000/api/chat/stream"
payload = {
    "message": "Write a 200 word article about the ocean",
    "thread_id": "debug-test-3"
}

print("Hitting the backend streaming endpoint — watch the timestamps:\n")
start = time.time()
line_count = 0

with requests.post(url, json=payload, stream=True) as resp:
    for raw_line in resp.iter_lines(decode_unicode=True):
        if raw_line and raw_line.startswith("data:"):
            line_count += 1
            elapsed = time.time() - start
            preview = raw_line[:80]
            print(f"[{elapsed:6.3f}s] line #{line_count}: {preview!r}")

total_time = time.time() - start
print(f"\nTotal SSE lines: {line_count}")
print(f"Total time: {total_time:.2f}s")

if line_count <= 3:
    print("\n⚠️  Very few lines received — the backend is NOT streaming")
    print("    progressively. Either the graph.py fix didn't take effect")
    print("    (did you restart uvicorn?), or something else is buffering")
    print("    the whole response before sending it.")
else:
    print(f"\n✅ Received {line_count} lines spread over {total_time:.2f}s —")
    print("   the backend IS streaming correctly. If the UI still shows")
    print("   everything at once, the issue is in the frontend's")
    print("   reader/decoder loop (ChatClient.tsx), not the backend.")