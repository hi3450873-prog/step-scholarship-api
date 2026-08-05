from flask import Flask, request, jsonify
from flask_cors import CORS

from step_matcher_mysql import ScholarshipMatcher

app = Flask(__name__)
CORS(app)

print("Loading scholarship matcher...")
matcher = ScholarshipMatcher()
matcher.fit()
print(f"Loaded {len(matcher.scholarships)} scholarships.")


@app.after_request
def add_security_headers(response):
    response.headers['Cross-Origin-Opener-Policy'] = 'same-origin-allow-popups'
    return response


@app.route("/")
def home():
    return jsonify({
        "status": "running",
        "message": "STEP Scholarship API is online"
    })


@app.route("/recommend", methods=["POST"])
def recommend():
    try:
        data = request.get_json()

        if not data:
            return jsonify({
                "success": False,
                "message": "No data received."
            }), 400

        # Retrieve top recommendations
        results = matcher.recommend(
            student=data,
            top_n=10,
            min_match=80,
            include_ineligible=False
        )

        print("\n========== API RESULTS ==========")
        print(results)
        print("Returned:", len(results))
        print("=================================\n")

        return jsonify({
            "results": results
        })

    except Exception as e:
        import traceback
        traceback.print_exc()

        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)