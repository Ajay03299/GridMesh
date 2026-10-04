"""Exercise the actual reused Streamlit view, every scenario and interval."""
from streamlit.testing.v1 import AppTest
from build_advisory_demo import SCENARIOS

app=AppTest.from_string('from src.visualization.advisory_demo import render_demo\nrender_demo()',default_timeout=30).run()
for name in SCENARIOS:
    app.selectbox[0].set_value(name).run()
    for interval in (0,2,4):
        app.slider[0].set_value(interval).run()
        assert not app.exception, [(e.message) for e in app.exception]
        assert len(app.metric)==4
        assert len(app.get('vega_lite_chart')) == 4
        assert any('Operator:' in m.value for m in app.markdown)
    print('PASS',name,'3 intervals, 4 advice metrics, operator response, no exception')
print('15 operator-view paths passed; no live control.')
