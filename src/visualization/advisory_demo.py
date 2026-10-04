"""Streamlit view for the deterministic decision-time-only advisory scenario."""
from html import escape

import altair as alt
import pandas as pd
import streamlit as st
from build_advisory_demo import SCENARIOS, scenario_demo
from src.visualization.dashboard_theme import palette


def planning_charts(d, colors):
    """Display decision-time inputs and allocations, never realized outcomes."""
    series = {'forecast_mw': 'Planning solar', 'grid_import_mw': 'Grid supply',
              'demand_mw': 'Demand'}
    supply = d[['time', *series]].rename(columns=series).melt(
        'time', var_name='Series', value_name='Power')
    supply['Power'] *= 1000
    supply_chart = alt.Chart(supply).mark_line(point=True, strokeWidth=2.5).encode(
        x=alt.X('time:T', title='Target time', axis=alt.Axis(format='%H:%M')),
        y=alt.Y('Power:Q', title='Power (kW)', scale=alt.Scale(zero=True)),
        color=alt.Color('Series:N', scale=alt.Scale(domain=list(series.values()),
            range=[colors['blue'], colors['muted'], colors['orange']]),
            legend=alt.Legend(orient='bottom', title=None)),
        tooltip=[alt.Tooltip('time:T', format='%H:%M'), 'Series:N',
                 alt.Tooltip('Power:Q', format='.1f')])

    allocation = pd.DataFrame({'time': d.time.repeat(2).to_numpy(),
        'Allocation': ['Scheduled backup', 'Planned uncovered reserve'] * len(d),
        'Power': d[['scheduled_backup_mw', 'planned_gap_mw']].to_numpy().ravel() * 1000})
    allocation_chart = alt.Chart(allocation).mark_bar(size=28).encode(
        x=alt.X('time:T', title='Target time', axis=alt.Axis(format='%H:%M')),
        y=alt.Y('Power:Q', title='Reserve power (kW)', stack='zero'),
        color=alt.Color('Allocation:N', scale=alt.Scale(
            domain=['Scheduled backup', 'Planned uncovered reserve'],
            range=[colors['blue'], colors['fault']]),
            legend=alt.Legend(orient='bottom', title=None)),
        order=alt.Order('Allocation:N', sort='descending'),
        tooltip=[alt.Tooltip('time:T', format='%H:%M'), 'Allocation:N',
                 alt.Tooltip('Power:Q', format='.1f')])

    budget = pd.DataFrame({'time': d.time,
        'Remaining energy': d.remaining_energy_before_mwh * 1000})
    budget_chart = alt.Chart(budget).mark_area(
        line=True, point=True, color=colors['blue'], opacity=.3).encode(
        x=alt.X('time:T', title='Target time', axis=alt.Axis(format='%H:%M')),
        y=alt.Y('Remaining energy:Q', title='Energy (kWh)', scale=alt.Scale(zero=True)),
        tooltip=[alt.Tooltip('time:T', format='%H:%M'),
                 alt.Tooltip('Remaining energy:Q', format='.1f')])

    def style(chart):
        return chart.properties(height=230).configure(background=colors['surface']).configure_axis(
            labelColor=colors['muted'], titleColor=colors['ink'], gridColor=colors['grid']
        ).configure_legend(labelColor=colors['ink']).configure_view(stroke=None)
    return tuple(style(chart) for chart in (supply_chart, allocation_chart, budget_chart))


def render_demo():
    colors = palette(st.session_state.get('dark_mode', False))
    st.html('<header class="gm-header"><div class="gm-kicker">GridMesh / operator workspace</div>'
            '<h1>Plan backup. Keep the shortfall visible.</h1>'
            '<p>Synthetic 100-home demo · +60-minute advice · generic backup<br>'
            'Advisory only. No equipment control or field validation.</p></header>')
    controls = st.columns([2, 1])
    name=controls[0].selectbox('Operator scenario',list(SCENARIOS),format_func=SCENARIOS.get)
    d, outcome = scenario_demo(name)
    ix = controls[1].slider('Target interval', 0, len(d)-1, 2)
    row = d.iloc[ix]
    with st.container(key='advice_snapshot'):
        st.html(f'<div class="gm-snapshot-title">{escape(SCENARIOS[name])} / target {row.time:%H:%M}</div>'
                f'<div class="gm-meta"><span>Issued <strong>{row.issued_at:%H:%M}</strong></span>'
                f'<span>Forecast age <strong>{row.forecast_age_minutes:.0f} min</strong></span>'
                f'<span>Planning solar <strong>{1000*row.forecast_mw:.0f} kW</strong></span></div>')
        cols = st.columns(4)
        cols[0].metric('Expected shortfall', f'{1000*row.expected_gap_mw:.0f} kW')
        cols[1].metric('Required reserve', f'{1000*row.required_backup_mw:.0f} kW')
        cols[2].metric('Scheduled backup advice', f'{1000*row.scheduled_backup_mw:.0f} kW')
        cols[3].metric('Planned uncovered reserve', f'{1000*row.planned_gap_mw:.0f} kW')
        st.html(f'<div class="gm-limits">Power limit <strong>{1000*row.backup_power_limit_mw:.0f} kW</strong>'
                f' &nbsp; / &nbsp; Remaining energy budget <strong>{1000*row.remaining_energy_before_mwh:.0f} kWh</strong>'
                ' before this decision<br>'
                f'Solar source: {escape(row.forecast_source.replace("_", " "))}. '
                f'Participating clients: {row.participating_clients}/100 (illustrative).</div>')
    if row.operator_attention:
        st.warning(f'{row.reason} Review required: refresh readings and confirm backup; '
                   'escalate uncovered reserve to the operator / DISCOM. A safety bound is not a forecast.')
    else:
        st.info(row.reason)
    st.markdown('<div class="gm-review"><strong>Operator:</strong> confirm backup availability and '
                'approve or reject advice. No command is sent.<br>'
                'Backup technology, metering, access and safety procedures require a pilot agreement.</div>',
                unsafe_allow_html=True)
    st.subheader('Reserve plan across target intervals')
    labels = {'required_backup_mw': 'Required reserve', 'scheduled_backup_mw': 'Scheduled backup',
              'planned_gap_mw': 'Planned uncovered reserve'}
    chart_data = d[['time', *labels]].rename(columns=labels).melt('time', var_name='Series', value_name='Power')
    chart_data['Power'] *= 1000
    chart = alt.Chart(chart_data).mark_line(point=True, strokeWidth=2.5).encode(
        x=alt.X('time:T', title='Target time', axis=alt.Axis(format='%H:%M')),
        y=alt.Y('Power:Q', title='Power (kW)'),
        color=alt.Color('Series:N', scale=alt.Scale(domain=list(labels.values()),
                        range=[colors['orange'], colors['blue'], colors['fault']]),
                        legend=alt.Legend(orient='bottom', title=None)),
        tooltip=['time:T', 'Series:N', alt.Tooltip('Power:Q', format='.1f')],
    ).properties(height=250).configure(background=colors['surface']).configure_axis(
        labelColor=colors['muted'], titleColor=colors['ink'], gridColor=colors['grid']
    ).configure_legend(labelColor=colors['ink']).configure_view(stroke=None)
    st.altair_chart(chart, width='stretch', theme=None)
    supply_chart, allocation_chart, budget_chart = planning_charts(d, colors)
    with st.container(key='planning_graphs'):
        st.subheader('What drives the backup decision?')
        left, right = st.columns(2)
        with left:
            st.markdown('**Solar, grid and demand**')
            st.altair_chart(supply_chart, width='stretch', theme=None)
            st.caption('Demand less grid supply and planning solar gives the expected gap. '
                       'Missing solar uses the existing zero-solar safety bound.')
        with right:
            st.markdown('**How much reserve can be covered?**')
            st.altair_chart(allocation_chart, width='stretch', theme=None)
            st.caption('Each bar totals required reserve. Blue is scheduled backup; '
                       'the remaining portion is planned uncovered reserve, not realized unmet energy.')
        st.markdown('**Remaining daily backup-energy budget**')
        st.altair_chart(budget_chart, width='stretch', theme=None)
        st.caption('Budget available before each decision. Scheduling backup consumes this allowance. '
                   'This is an assumed scheduling budget, not measured battery state of charge.')
    with st.expander('Ex-post synthetic outcome — not a decision input'):
        st.dataframe(d[['time','actual_mw','actual_delivered_backup_mw','realized_unserved_mwh']], hide_index=True)
        st.write(f'Simulated fully supplied intervals: {outcome["availability_pct"]:.1f}%. '
                 f'Realized unserved energy: {1000*outcome["ens_mwh"]:.1f} kWh. '
                 f'Scheduled backup: {1000*outcome["backup_mwh"]:.1f} kWh.')
        st.caption('Realized ENS is evaluated using actual solar after the target interval. Planned uncovered reserve is not ENS.')
