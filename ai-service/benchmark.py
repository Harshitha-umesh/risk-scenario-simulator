import requests
import time

# 🔗 Endpoints
QUERY_URL = "http://127.0.0.1:5000/query"
GENERATE_URL = "http://127.0.0.1:5000/generate-report"
STATUS_URL = "http://127.0.0.1:5000/report-status/{}"

# 🧮 Percentile function
def percentile(data, percent):
    data = sorted(data)
    k = int(len(data) * percent)
    return data[k]

# 🔹 Test QUERY API (slow → AI call)
def test_query():
    times = []

    for i in range(20):  # keep small to avoid overload
        start = time.time()

        res = requests.post(
            QUERY_URL,
            json={"question": f"cyber attack risk {i}"}
        )

        end = time.time()

        if res.status_code == 200:
            times.append((end - start) * 1000)

        time.sleep(1)  # ✅ avoid overload

    print("\nQUERY RESULTS:")
    print("p50:", percentile(times, 0.50))
    print("p95:", percentile(times, 0.95))
    print("p99:", percentile(times, 0.99))


# 🔹 Test GENERATE REPORT (fast → async)
def test_generate():
    times = []

    for i in range(20):
        start = time.time()

        res = requests.post(
            GENERATE_URL,
            json={"question": f"cyber attack risk {i}"}
        )

        end = time.time()

        if res.status_code == 200:
            times.append((end - start) * 1000)

    print("\nGENERATE REPORT RESULTS:")
    print("p50:", percentile(times, 0.50))
    print("p95:", percentile(times, 0.95))
    print("p99:", percentile(times, 0.99))


# 🔹 Test STATUS API (very fast)
def test_status():
    times = []

    # First create one job
    res = requests.post(
        GENERATE_URL,
        json={"question": "cyber attack risk"}
    )

    job_id = res.json().get("job_id")

    # wait a bit so job exists
    time.sleep(2)

    for i in range(20):
        start = time.time()

        res = requests.get(STATUS_URL.format(job_id))

        end = time.time()

        if res.status_code == 200:
            times.append((end - start) * 1000)

    print("\nREPORT STATUS RESULTS:")
    print("p50:", percentile(times, 0.50))
    print("p95:", percentile(times, 0.95))
    print("p99:", percentile(times, 0.99))


#  RUN ALL
if __name__ == "__main__":
    test_query()
    test_generate()
    test_status()
    import requests
import time

# 🔗 Endpoints
QUERY_URL = "http://127.0.0.1:5000/query"
GENERATE_URL = "http://127.0.0.1:5000/generate-report"
STATUS_URL = "http://127.0.0.1:5000/report-status/{}"

# 🧮 Percentile function
def percentile(data, percent):
    data = sorted(data)
    k = int(len(data) * percent)
    return data[k]

# 🔹 Test QUERY API (slow → AI call)
def test_query():
    times = []

    for i in range(20):  # keep small to avoid overload
        start = time.time()

        res = requests.post(
            QUERY_URL,
            json={"question": f"cyber attack risk {i}"}
        )

        end = time.time()

        if res.status_code == 200:
            times.append((end - start) * 1000)

        time.sleep(1)  # ✅ avoid overload

    print("\nQUERY RESULTS:")
    print("p50:", percentile(times, 0.50))
    print("p95:", percentile(times, 0.95))
    print("p99:", percentile(times, 0.99))


# 🔹 Test GENERATE REPORT (fast → async)
def test_generate():
    times = []

    for i in range(20):
        start = time.time()

        res = requests.post(
            GENERATE_URL,
            json={"question": f"cyber attack risk {i}"}
        )

        end = time.time()

        if res.status_code == 200:
            times.append((end - start) * 1000)

    print("\nGENERATE REPORT RESULTS:")
    print("p50:", percentile(times, 0.50))
    print("p95:", percentile(times, 0.95))
    print("p99:", percentile(times, 0.99))


# 🔹 Test STATUS API (very fast)
def test_status():
    times = []

    # First create one job
    res = requests.post(
        GENERATE_URL,
        json={"question": "cyber attack risk"}
    )

    job_id = res.json().get("job_id")

    # wait a bit so job exists
    time.sleep(2)

    for i in range(20):
        start = time.time()

        res = requests.get(STATUS_URL.format(job_id))

        end = time.time()

        if res.status_code == 200:
            times.append((end - start) * 1000)

    print("\nREPORT STATUS RESULTS:")
    print("p50:", percentile(times, 0.50))
    print("p95:", percentile(times, 0.95))
    print("p99:", percentile(times, 0.99))


#  RUN ALL
if __name__ == "__main__":
    test_query()
    test_generate()
    test_status()