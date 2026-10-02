"""Headless dashboard regression smoke test after building replay data."""
from streamlit.testing.v1 import AppTest


def main():
    app = AppTest.from_file("dashboard.py", default_timeout=90).run()
    snapshots = {}
    for dark in (True, False):
        app.sidebar.toggle[0].set_value(dark).run()
        for scenario in ("normal", "faulty", "dropout", "shift"):
            app.sidebar.radio[0].set_value(scenario).run()
            if app.exception:
                raise AssertionError(f"{scenario}: {[e.message for e in app.exception]}")
            assert app.sidebar.toggle[0].value == dark
            assert len(app.metric) >= 10, f"{scenario}: missing operator metrics"
            metrics = [(m.label, m.value, m.delta) for m in app.metric]
            if dark:
                snapshots[scenario] = metrics
            else:
                assert metrics == snapshots[scenario], "Theme changed simulation metrics"
            print(f"PASS dashboard replay: {scenario}, {'dark' if dark else 'light'}")


if __name__ == "__main__":
    main()
