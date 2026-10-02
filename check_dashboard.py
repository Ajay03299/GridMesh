"""Headless dashboard regression smoke test after building replay data."""
from streamlit.testing.v1 import AppTest


def main():
    app = AppTest.from_file("dashboard.py", default_timeout=90).run()
    for scenario in ("normal", "faulty", "dropout", "shift"):
        app.sidebar.radio[0].set_value(scenario).run()
        if app.exception:
            raise AssertionError(f"{scenario}: {[e.message for e in app.exception]}")
        assert len(app.metric) >= 10, f"{scenario}: missing operator metrics"
        print(f"PASS dashboard replay: {scenario}")


if __name__ == "__main__":
    main()
