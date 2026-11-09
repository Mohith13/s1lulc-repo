"""Self-contained interactive HTML report for one prototype run.

One file, no server: map images and charts are embedded as data URIs. The month slider and
layer switches are plain JavaScript. The optional basemap view loads Leaflet and
OpenStreetMap tiles from the internet; everything else works offline.
"""

from __future__ import annotations

import base64
import html
import json
from pathlib import Path

import numpy as np

# Leaflet 1.9.4 (BSD-2-Clause, see assets/LEAFLET_LICENSE) is embedded so the report does not
# depend on a CDN; only the basemap tiles need internet access.
_ASSETS = Path(__file__).parent / "assets"


def _asset(name: str) -> str:
    return (_ASSETS / name).read_text(encoding="utf-8")


def data_uri(png: bytes) -> str:
    return "data:image/png;base64," + base64.b64encode(png).decode("ascii")


def _fmt(x, nd=1, sign=False):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "–"
    return f"{x:+.{nd}f}" if sign else f"{x:.{nd}f}"


def _cls(name: str) -> str:
    return html.escape(name.replace("_", " "))


def _swatch(color: str) -> str:
    return f'<span class="sw" style="background:{color}"></span>'


def build_html(res: dict, ctx: dict, layers: dict[str, list[str]], charts: dict[str, str]) -> str:
    """res: analyse() output; ctx: run metadata; layers: name -> 12 data URIs; charts: data URIs."""
    names, colors = res["class_names"], ctx["class_colors"]
    months = ctx["months"]
    raw, sm = res["series"]["raw"], res["series"]["smoothed"]
    f, l_ = raw["first_month"], raw["last_month"]
    has_sig = "significant_total" in raw
    has_adj = "adjusted" in raw
    dw = res.get("dynamic_world")

    # --- summary cards -------------------------------------------------------------------
    cards = []
    for c, n in enumerate(names):
        d = raw["total"][c]
        sig = bool(raw["significant_total"][c]) if has_sig else None
        badge = ("" if sig is None else
                 '<span class="badge sig">significant</span>' if sig else
                 '<span class="badge noise">within noise</span>')
        cards.append(
            f'<div class="card">{_swatch(colors[c])}<div class="cname">{_cls(n)}</div>'
            f'<div class="big">{_fmt(raw["shares"][l_, c])}%</div>'
            f'<div class="sub">{_fmt(raw["shares"][f, c])}% → {_fmt(raw["shares"][l_, c])}% '
            f'(<b>{_fmt(d, 1, True)} pp</b>)</div>{badge}</div>')

    # --- tables --------------------------------------------------------------------------
    head = "".join(f"<th>{_swatch(colors[c])}{_cls(n)}</th>" for c, n in enumerate(names))
    share_rows = []
    for i, m in enumerate(months):
        cells = []
        for c in range(len(names)):
            v = raw["shares"][i, c]
            extra = ""
            if has_adj and not np.isnan(raw["adjusted"][i, c]):
                extra = (f'<div class="adj">adj. {_fmt(raw["adjusted"][i, c])} ± '
                         f'{_fmt(raw["adjusted_ci95"][i, c])}</div>')
            cells.append(f"<td>{_fmt(v)}{extra}</td>")
        share_rows.append(f"<tr><th>{m}</th>{''.join(cells)}</tr>")

    def change_table(series, key_sig, key_vals, row_labels):
        rows = []
        vals = series[key_vals]
        for i, lab in enumerate(row_labels):
            cells = []
            for c in range(len(names)):
                v = vals[i, c]
                s = series.get(key_sig)
                cls = ""
                if s is not None and not np.isnan(v):
                    cls = ' class="sig"' if s[i, c] else ' class="noise"'
                cells.append(f"<td{cls}>{_fmt(v, 2, True)}</td>")
            rows.append(f"<tr><th>{lab}</th>{''.join(cells)}</tr>")
        return "".join(rows)

    pairs = [f"{a} → {b}" for a, b in zip(months[:-1], months[1:], strict=True)]
    m2m_raw = change_table(raw, "significant_m2m", "month_to_month", pairs)
    m2m_sm = change_table(sm, "significant_m2m", "month_to_month", pairs)

    total_rows = []
    for c, n in enumerate(names):
        floor = _fmt(raw["floor_total"][c], 2) if has_sig else "–"
        sig = ("–" if not has_sig else "yes" if raw["significant_total"][c] else "no")
        total_rows.append(
            f"<tr><th>{_swatch(colors[c])}{_cls(n)}</th><td>{_fmt(raw['shares'][f, c])}</td>"
            f"<td>{_fmt(raw['shares'][l_, c])}</td><td><b>{_fmt(raw['total'][c], 2, True)}</b></td>"
            f"<td>{floor}</td><td>{sig}</td><td>{_fmt(sm['total'][c], 2, True)}</td>"
            f"<td>{_fmt(raw['gross_gain'][c], 2)}</td>"
            f"<td>{_fmt(raw['gross_loss'][c], 2)}</td></tr>")

    trans_rows = "".join(
        f"<tr><th>{_cls(names[i])}</th>" + "".join(
            f'<td{" class=diag" if i == j else ""}>{_fmt(raw["transitions"][i, j], 2)}</td>'
            for j in range(len(names))) + "</tr>" for i in range(len(names)))

    warn_html = "".join(
        f'<li><b>{html.escape(w["kind"].replace("_", " "))}</b>'
        f'{" (" + ", ".join(months[i] for i in w["months"]) + ")" if w["months"] else ""}: '
        f'{html.escape(w["text"])}</li>' for w in res["warnings"])

    acq_html = ""
    if res.get("platforms") is not None:
        pm = res.get("passes_median") or [None] * len(months)
        sc = res.get("scenes") or [None] * len(months)
        dwi = ctx.get("dw_images") or [None] * len(months)
        sw = {i for w in res["warnings"] if w["kind"] == "sensor_change" for i in w["months"]}
        rows = "".join(
            f'<tr><th>{m}</th><td>{"–" if sc[i] is None else sc[i]}</td>'
            f'<td>{"–" if pm[i] is None else pm[i]}</td>'
            f'<td{" class=sig" if i in sw else ""}>'
            f'{" + ".join("S1" + p for p in res["platforms"][i]) or "none"}</td>'
            f'<td>{"–" if dwi[i] is None else dwi[i]}</td></tr>' for i, m in enumerate(months))
        acq_html = ("<h2>Data provenance</h2><div class=tw><table><tr><th>month</th>"
                    "<th>Sentinel-1 scenes</th><th>median passes per pixel</th>"
                    "<th>satellites</th><th>Dynamic World images</th></tr>" + rows +
                    "</table></div><p class=notes>Highlighted: the satellite combination "
                    "differs from the previous month.</p>")
    dw_line = ""
    if dw and dw["mean_agreement"] is not None:
        dw_line = (f'<div class="kpi"><div class="big">{100 * dw["mean_agreement"]:.0f}%</div>'
                   '<div class="sub">mean monthly pixel agreement with Dynamic World</div></div>')
    unc = float(np.nanmean(res["maps"]["entropy_mean"]))

    layer_names = list(layers)
    left_opts = "".join(f'<option value="{k}">{html.escape(k)}</option>' for k in layer_names)
    right_default = "Dynamic World" if "Dynamic World" in layers else layer_names[-1]
    right_opts = "".join(f'<option value="{k}"{" selected" if k == right_default else ""}>'
                         f'{html.escape(k)}</option>' for k in layer_names)
    legend ="".join(f"<span>{_swatch(colors[c])}{_cls(n)}</span>" for c, n in enumerate(names))
    chart_html = "".join(f'<figure><img src="{uri}" alt="{html.escape(k)}">'
                         f'<figcaption>{html.escape(k)}</figcaption></figure>'
                         for k, uri in charts.items())
    b = ctx["bounds_wgs84"]
    sig_note = ("Changes are flagged <b>significant</b> when their magnitude exceeds the model's "
                f"noise floor ({ctx.get('noise_quantile', 95):.0f}th percentile of change "
                "measured on stable validation/test regions)." if has_sig else
                "No noise floor in this model bundle: significance flags unavailable.")

    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Land cover change report</title>
<style>{_asset("leaflet.css")}</style>
<style>
:root{{--bg:#f7f6f2;--card:#fff;--ink:#1d1d1b;--mut:#6b6a64;--line:#e3e1d9;--acc:#1f5f8b;
--sig:#fde2d9;--sigi:#8a2b0e;--noi:#eceae3;--noii:#6b6a64}}
@media (prefers-color-scheme:dark){{:root{{--bg:#1b1b19;--card:#262624;--ink:#ecebe6;
--mut:#a3a19a;--line:#3a3935;--acc:#7fb5dd;--sig:#5a2a1c;--sigi:#ffd2c2;--noi:#33322e;
--noii:#b5b3ab}}}}
*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--ink);
font:15px/1.55 system-ui,-apple-system,"Segoe UI",sans-serif}}
main{{max-width:1180px;margin:0 auto;padding:28px 20px 60px}}
h1{{font-size:26px;margin:0 0 4px;font-weight:600}}h2{{font-size:19px;margin:36px 0 10px;
font-weight:600}}.meta{{color:var(--mut)}}
.grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(150px,1fr));gap:10px}}
.card,.kpi,.panel{{background:var(--card);border:1px solid var(--line);border-radius:10px;
padding:12px}}.cname{{font-size:13px;color:var(--mut);display:inline}}
.big{{font-size:22px;font-weight:600;margin-top:2px}}.sub{{font-size:12.5px;color:var(--mut)}}
.badge{{display:inline-block;margin-top:6px;font-size:11.5px;padding:1px 8px;border-radius:9px}}
.badge.sig{{background:var(--sig);color:var(--sigi)}}.badge.noise{{background:var(--noi);
color:var(--noii)}}.sw{{display:inline-block;width:11px;height:11px;border-radius:2px;
margin-right:6px;vertical-align:-1px;border:1px solid rgba(0,0,0,.15)}}
.kpis{{display:grid;grid-template-columns:repeat(auto-fill,minmax(220px,1fr));gap:10px;
margin-top:10px}}.warn{{background:var(--sig);color:var(--sigi);border-radius:10px;
padding:10px 16px;margin-top:16px}}.warn ul{{margin:4px 0;padding-left:18px}}
.viewer{{display:grid;grid-template-columns:1fr 1fr;gap:12px}}
.viewer img{{width:100%;image-rendering:pixelated;border-radius:6px;display:block;
background:repeating-conic-gradient(var(--line) 0 25%,transparent 0 50%) 0 0/16px 16px}}
.controls{{display:flex;flex-wrap:wrap;gap:12px;align-items:center;margin:8px 0 12px}}
input[type=range]{{flex:1;min-width:200px}}select,button{{font:inherit;padding:4px 8px;
border-radius:6px;border:1px solid var(--line);background:var(--card);color:var(--ink)}}
#mlabel{{font-weight:600;min-width:70px}}.legend{{display:flex;flex-wrap:wrap;gap:12px;
font-size:13px;color:var(--mut)}}
#map{{height:460px;border-radius:10px;border:1px solid var(--line)}}
.tw{{overflow-x:auto}}table{{border-collapse:collapse;width:100%;background:var(--card);
font-size:13px;font-variant-numeric:tabular-nums}}th,td{{border-bottom:1px solid var(--line);
padding:5px 8px;text-align:right;white-space:nowrap}}th{{text-align:left;font-weight:600}}
td.sig{{background:var(--sig);color:var(--sigi);font-weight:600}}td.noise{{color:var(--mut)}}
td.diag{{font-weight:600}}.adj{{font-size:11px;color:var(--mut)}}
figure{{margin:14px 0;background:var(--card);border:1px solid var(--line);border-radius:10px;
padding:10px}}figure img{{width:100%}}figcaption{{font-size:12.5px;color:var(--mut)}}
.tabs button.on{{background:var(--acc);color:#fff;border-color:var(--acc)}}
.notes{{font-size:13.5px;color:var(--mut);max-width:880px}}
@media (max-width:720px){{.viewer{{grid-template-columns:1fr}}}}
</style></head><body><main>
<h1>Land cover change report</h1>
<div class="meta">{ctx['lat']:.5f}° N, {ctx['lon']:.5f}° E · {ctx['area_km2']:.0f} km² ·
{months[0]} to {months[-1]} · model <code>{html.escape(ctx['variant'])}</code> ·
Sentinel-1 only · generated {html.escape(ctx['generated'])}</div>
{f'<div class="warn"><b>Data quality notes</b><ul>{warn_html}</ul></div>' if warn_html else ''}

<h2>Total change, {months[f]} → {months[l_]}</h2>
<div class="grid">{''.join(cards)}</div>
<div class="kpis">
<div class="kpi"><div class="big">{raw['changed_total']:.1f}%</div><div class="sub">of the area
has a different label in the last month than in the first (gross change)</div></div>
<div class="kpi"><div class="big">{raw['flicker_mean']:.2f}</div><div class="sub">label switches
per pixel per year (raw); {sm['flicker_mean']:.2f} after temporal smoothing</div></div>
<div class="kpi"><div class="big">{unc:.2f}</div><div class="sub">mean uncertainty (0 = certain,
1 = uniform over classes)</div></div>{dw_line}</div>

<h2>Monthly maps</h2>
<div class="controls"><button id="play">▶ play</button>
<input id="month" type="range" min="0" max="{len(months) - 1}" value="0"><span id="mlabel"></span>
<label>left <select id="left">{left_opts}</select></label>
<label>right <select id="right">{right_opts}</select></label></div>
<div class="viewer"><div><img id="imgL" alt="left layer"></div><div><img id="imgR"
alt="right layer"></div></div>
<div class="legend" style="margin-top:8px">{legend}</div>

<h2>On the map</h2>
<div class="controls"><label>layer <select id="mlayer">{left_opts}</select></label>
<label>opacity <input id="op" type="range" min="0" max="100" value="75"></label>
<span class="sub">follows the month slider · basemap: Esri satellite or CARTO street map, switch top right (needs internet)
</span></div><div id="map"></div>

<h2>Class shares per month (%)</h2>
<div class="tw"><table><tr><th>month</th>{head}</tr>{''.join(share_rows)}</table></div>
<p class="notes">Shares are computed on the area that has radar data in every month
({100 * res['common_valid_fraction']:.1f}% of the area of interest), so missing data cannot appear
as change.{' "adj." = error-adjusted share with 95% confidence interval.' if has_adj else ''}</p>

<h2>Month-to-month change (percentage points)</h2>
<div class="tabs controls"><button class="on" data-t="t_raw">raw maps</button>
<button data-t="t_sm">temporally smoothed</button></div>
<div class="tw"><table id="t_raw"><tr><th>months</th>{head}</tr>{m2m_raw}</table>
<table id="t_sm" style="display:none"><tr><th>months</th>{head}</tr>{m2m_sm}</table></div>
<p class="notes">Highlighted cells exceed the noise floor; grey cells are within the model's own
month-to-month noise. {sig_note}</p>

<h2>Total change and gross vs net</h2>
<div class="tw"><table><tr><th>class</th><th>{months[f]} %</th><th>{months[l_]} %</th>
<th>change pp</th><th>noise floor pp</th><th>significant</th><th>smoothed pp</th>
<th>gross gain pp</th><th>gross loss pp</th></tr>{''.join(total_rows)}</table></div>

<h2>Transitions, {months[f]} → {months[l_]} (% of area)</h2>
<div class="tw"><table><tr><th>from \\ to</th>{head}</tr>{trans_rows}</table></div>

<h2>Charts</h2>{chart_html}
{acq_html}

<h2>How to read this report</h2>
<div class="notes"><p><b>Maps.</b> Each month is classified independently by a U-Net from a
monthly mean of Sentinel-1 VV/VH backscatter. "Smoothed" averages class probabilities over
neighbouring months ({res['smoothing_window']}-month window) before choosing a class, which
suppresses one-month flips caused by seasonal confusion.</p>
<p><b>Changes.</b> Month-to-month change = share in month m minus share in month m−1; total change
= last month minus first month; both in percentage points. Net change can hide large swaps:
gross gain/loss show how much area entered and left each class.</p>
<p><b>Significance.</b> {sig_note} A change below the floor may be pure classification
noise, including seasonal effects such as bare soil after harvest looking like another class.</p>
<p><b>Error-adjusted shares</b> correct the mapped shares with the model's validation confusion
matrix (Olofsson et al., 2014). The interval assumes {ctx.get('effective_n', 500)} effectively
independent reference samples per class and that validation errors transfer to this location.</p>
<p><b>Dynamic World</b> is an independent Sentinel-2-based map (never used for training), shown as a
monthly mode of cloud-free scenes. It is itself a model output, so agreement is a plausibility
check, not an accuracy measurement.</p></div>
</main>
<script>
const LAYERS = {json.dumps(layers)};
const MONTHS = {json.dumps(months)};
const B = [[{b[1]}, {b[0]}], [{b[3]}, {b[2]}]];
const $ = id => document.getElementById(id);
let map = null, overlay = null;
function draw() {{
  const i = +$('month').value;
  $('mlabel').textContent = MONTHS[i];
  $('imgL').src = LAYERS[$('left').value][i];
  $('imgR').src = LAYERS[$('right').value][i];
  if (overlay) {{ overlay.setUrl(LAYERS[$('mlayer').value][i]);
                  overlay.setOpacity($('op').value / 100); }}
}}
['month', 'left', 'right', 'mlayer', 'op'].forEach(id => $(id).addEventListener('input', draw));
let timer = null;
$('play').onclick = () => {{
  if (timer) {{ clearInterval(timer); timer = null; $('play').textContent = '▶ play'; return; }}
  $('play').textContent = '❚❚ pause';
  timer = setInterval(() => {{ $('month').value = (+$('month').value + 1) % MONTHS.length;
                               draw(); }}, 900);
}};
document.querySelectorAll('.tabs button').forEach(btn => btn.onclick = () => {{
  document.querySelectorAll('.tabs button').forEach(x => x.classList.remove('on'));
  btn.classList.add('on');
  ['t_raw', 't_sm'].forEach(t => $(t).style.display = t === btn.dataset.t ? '' : 'none');
}});
draw();
</script>
<script>{_asset("leaflet.js")}</script>
<script>
if (!window.L) {{ document.getElementById('map').textContent = 'Map library failed to load.'; }}
if (window.L) {{
  map = L.map('map').fitBounds(B);
  // OpenStreetMap's own tile servers refuse requests without a Referer header, which is
  // what a report opened from disk sends; these providers serve standalone pages.
  const sat = L.tileLayer(
    'https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{{z}}/{{y}}/{{x}}',
    {{maxZoom: 19, attribution: 'Tiles &copy; Esri &mdash; Source: Esri, Maxar, Earthstar Geographics, and the GIS User Community'}});
  const street = L.tileLayer('https://{{s}}.basemaps.cartocdn.com/light_all/{{z}}/{{x}}/{{y}}{{r}}.png',
    {{maxZoom: 19, subdomains: 'abcd', attribution: '&copy; OpenStreetMap contributors &copy; CARTO'}});
  sat.addTo(map);
  L.control.layers({{'Satellite (Esri)': sat, 'Street map (CARTO)': street}}, null,
                   {{collapsed: false}}).addTo(map);
  overlay = L.imageOverlay(LAYERS[$('mlayer').value][0], B, {{opacity: 0.75}}).addTo(map);
  L.rectangle(B, {{color: '#1f5f8b', weight: 1, fill: false}}).addTo(map);
  draw();
}}
</script>
</body></html>"""
