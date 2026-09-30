"""
SpectraSense - Structured Signal Profile Generator
Produces the official audit trail and intelligence report in JSON and HTML format.
Standardized Automated Signal Parameter Extraction & Modulation Analysis.
"""

import json
import datetime


def build_signal_profile(file_meta, preprocess_meta, features, hypotheses, validation_result):
    """
    Constructs a comprehensive, standardized Structured Signal Profile dictionary.
    """
    timestamp = datetime.datetime.now(datetime.timezone.utc).isoformat()
    
    top_candidate = hypotheses[0]["modulation"] if hypotheses else "Unknown"
    top_confidence = hypotheses[0]["confidence_pct"] if hypotheses else 0.0
    
    profile = {
        "metadata": {
            "application": "SpectraSense Workflow Orchestration & Evidence Layer",
            "standard": "Automated Signal Parameter Extraction & Modulation Analysis",
            "timestamp_utc": timestamp,
            "profile_id": f"SIG-PRF-{int(datetime.datetime.now().timestamp())}",
            "source_file": file_meta.get("filename", "unknown"),
            "file_size_bytes": file_meta.get("size_bytes", 0),
            "file_format": file_meta.get("format_guess", "IQ Float32")
        },
        "characterization": {
            "carrier_frequency_hz": features.get("carrier_freq_hz", 0),
            "carrier_centroid_hz": features.get("carrier_centroid_hz", 0),
            "bandwidth_3db_hz": features.get("bandwidth_3db_hz", 0),
            "occupied_bandwidth_99_hz": features.get("bandwidth_99_hz", 0),
            "estimated_snr_db": features.get("snr_db", 0),
            "symbol_rate_baud": features.get("symbol_rate_baud", 0),
            "samples_per_symbol": features.get("samples_per_symbol", 0),
            "cumulant_c40": features.get("cumulant_c40", 0),
            "cumulant_c42": features.get("cumulant_c42", 0),
            "envelope_variance": features.get("envelope_variance", 0),
            "inst_freq_variance": features.get("freq_inst_variance", 0)
        },
        "hypotheses_ranked": hypotheses,
        "processing_and_validation": {
            "top_candidate": top_candidate,
            "top_candidate_confidence_pct": top_confidence,
            "validation_status": validation_result.get("status", "PENDING"),
            "validation_verdict": validation_result.get("verdict", "N/A"),
            "evm_percent": validation_result.get("evm_percent"),
            "refinement_cycles_used": validation_result.get("refinement_count", 0),
            "processing_pipeline": validation_result.get("processing_path_used", "N/A"),
            "uncertainty_notes": validation_result.get("uncertainty_notes", [])
        },
        "orchestration_summary": {
            "evidence_chain_complete": True,
            "verdict_is_conclusive": validation_result.get("passed", False),
            "requires_human_sigint_review": not validation_result.get("passed", False)
        }
    }
    return profile


def export_profile_json(profile, filepath):
    """Exports profile to a formatted JSON file."""
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(profile, f, indent=2)


def generate_profile_html(profile):
    """
    Generates a clean, professional, print/PDF-ready HTML report with dark intelligence styling.
    """
    meta = profile["metadata"]
    char = profile["characterization"]
    val = profile["processing_and_validation"]
    hypotheses = profile["hypotheses_ranked"]
    
    status_color = "#10b981" if val["validation_status"].startswith("VALIDATED") else ("#f59e0b" if "REFINEMENT" in val["validation_status"] else "#ef4444")

    rows_html = ""
    for h in hypotheses:
        evidence_pills = "".join([f"<span class='badge'>{e}</span>" for e in h.get("evidence", [])])
        rows_html += f"""
        <tr>
            <td><strong>#{h['rank']}</strong></td>
            <td><strong style="color: #00d2ff;">{h['modulation']}</strong></td>
            <td>
                <div style="display: flex; align-items: center; gap: 8px;">
                    <div style="flex: 1; height: 8px; background: #2d3748; border-radius: 4px; overflow: hidden;">
                        <div style="width: {h['confidence_pct']}%; height: 100%; background: #00d2ff;"></div>
                    </div>
                    <span>{h['confidence_pct']}%</span>
                </div>
            </td>
            <td>{evidence_pills}</td>
            <td style="font-size: 11px; color: #a0aec0;">{h.get('suggested_pipeline', '-')}</td>
        </tr>
        """

    uncertainty_items = "".join([f"<li>{note}</li>" for note in val.get("uncertainty_notes", [])])

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>SpectraSense Profile - {meta['profile_id']}</title>
<style>
    body {{
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
        background-color: #0d1117;
        color: #e2e8f0;
        margin: 0;
        padding: 32px;
        line-height: 1.5;
    }}
    .container {{
        max-width: 960px;
        margin: 0 auto;
        background-color: #161b22;
        border: 1px solid #30363d;
        border-radius: 8px;
        padding: 32px;
    }}
    .header {{
        display: flex;
        justify-content: space-between;
        align-items: flex-start;
        border-bottom: 1px solid #30363d;
        padding-bottom: 20px;
        margin-bottom: 24px;
    }}
    .title {{
        font-size: 22px;
        font-weight: 700;
        letter-spacing: 0.5px;
        color: #f0f6fc;
        margin: 0 0 4px 0;
    }}
    .subtitle {{
        font-size: 13px;
        color: #8b949e;
        margin: 0;
    }}
    .status-box {{
        padding: 8px 16px;
        border-radius: 6px;
        font-size: 13px;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 1px;
        border: 1px solid {status_color};
        color: {status_color};
        background: rgba(0,0,0,0.3);
    }}
    .grid-2 {{
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 16px;
        margin-bottom: 24px;
    }}
    .card {{
        background-color: #0d1117;
        border: 1px solid #30363d;
        border-radius: 6px;
        padding: 16px;
    }}
    .card-title {{
        font-size: 12px;
        text-transform: uppercase;
        letter-spacing: 1px;
        color: #8b949e;
        margin-bottom: 12px;
        font-weight: 600;
    }}
    .param-table {{
        width: 100%;
        border-collapse: collapse;
        font-size: 13px;
    }}
    .param-table td {{
        padding: 6px 0;
        border-bottom: 1px solid #21262d;
    }}
    .param-label {{
        color: #8b949e;
    }}
    .param-val {{
        text-align: right;
        font-family: "SF Mono", "Consolas", monospace;
        font-weight: 600;
        color: #58a6ff;
    }}
    table.data-table {{
        width: 100%;
        border-collapse: collapse;
        font-size: 13px;
        margin-top: 12px;
    }}
    table.data-table th {{
        text-align: left;
        padding: 10px;
        background: #21262d;
        color: #8b949e;
        font-weight: 600;
        border-bottom: 2px solid #30363d;
    }}
    table.data-table td {{
        padding: 10px;
        border-bottom: 1px solid #21262d;
    }}
    .badge {{
        display: inline-block;
        padding: 2px 6px;
        font-size: 11px;
        background: #21262d;
        border: 1px solid #30363d;
        border-radius: 4px;
        margin-right: 4px;
        margin-bottom: 4px;
        color: #c9d1d9;
    }}
    .uncertainty-box {{
        background: #1f1a14;
        border: 1px solid #d97706;
        border-radius: 6px;
        padding: 16px;
        margin-top: 24px;
    }}
    .uncertainty-title {{
        color: #f59e0b;
        font-weight: 700;
        font-size: 13px;
        margin-bottom: 8px;
    }}
    ul {{
        margin: 0;
        padding-left: 20px;
        font-size: 13px;
        color: #d1d5db;
    }}
    .footer {{
        margin-top: 32px;
        border-top: 1px solid #30363d;
        padding-top: 16px;
        font-size: 11px;
        color: #6e7681;
        display: flex;
        justify-content: space-between;
    }}
</style>
</head>
<body>
<div class="container">
    <div class="header">
        <div>
            <h1 class="title">SPECTRASENSE STRUCTURED SIGNAL PROFILE</h1>
            <p class="subtitle">Signal Workflow Orchestration &bull; Evidence Record</p>
        </div>
        <div class="status-box">{val['validation_status']}</div>
    </div>

    <div class="grid-2">
        <div class="card">
            <div class="card-title">Ingest Metadata</div>
            <table class="param-table">
                <tr><td class="param-label">Profile ID</td><td class="param-val">{meta['profile_id']}</td></tr>
                <tr><td class="param-label">Source File</td><td class="param-val">{meta['source_file']}</td></tr>
                <tr><td class="param-label">Format</td><td class="param-val">{meta['file_format']}</td></tr>
                <tr><td class="param-label">Timestamp</td><td class="param-val">{meta['timestamp_utc']}</td></tr>
            </table>
        </div>

        <div class="card">
            <div class="card-title">Extracted Physical DSP Parameters</div>
            <table class="param-table">
                <tr><td class="param-label">Carrier Freq (Fc)</td><td class="param-val">{char['carrier_frequency_hz']:,.1f} Hz</td></tr>
                <tr><td class="param-label">3 dB Bandwidth</td><td class="param-val">{char['bandwidth_3db_hz']:,.1f} Hz</td></tr>
                <tr><td class="param-label">Estimated SNR</td><td class="param-val">{char['estimated_snr_db']} dB</td></tr>
                <tr><td class="param-label">Symbol Rate (Rs)</td><td class="param-val">{char['symbol_rate_baud']:,.1f} Baud</td></tr>
            </table>
        </div>
    </div>

    <div class="card" style="margin-bottom: 24px;">
        <div class="card-title">Ranked Modulation Hypotheses (Stage 4)</div>
        <table class="data-table">
            <thead>
                <tr>
                    <th style="width: 60px;">Rank</th>
                    <th style="width: 120px;">Candidate</th>
                    <th style="width: 180px;">Confidence</th>
                    <th>Physics-Informed Evidence Drivers</th>
                    <th>Recommended Processing Chain</th>
                </tr>
            </thead>
            <tbody>
                {rows_html}
            </tbody>
        </table>
    </div>

    <div class="card">
        <div class="card-title">Validation &amp; Verification Verdict (Stage 5)</div>
        <table class="param-table">
            <tr><td class="param-label">Top Candidate</td><td class="param-val">{val['top_candidate']} ({val['top_candidate_confidence_pct']}%)</td></tr>
            <tr><td class="param-label">EVM Residual</td><td class="param-val">{str(val['evm_percent']) + '%' if val['evm_percent'] is not None else 'N/A'}</td></tr>
            <tr><td class="param-label">Refinement Loops</td><td class="param-val">{val['refinement_cycles_used']} cycles</td></tr>
            <tr><td class="param-label">Validation Verdict</td><td class="param-val" style="color: {status_color};">{val['validation_verdict']}</td></tr>
        </table>
    </div>

    <div class="uncertainty-box">
        <div class="uncertainty-title">AUDIT TRAIL &amp; UNCERTAINTY NOTES (BOUNDED REFINEMENT EVIDENCE)</div>
        <ul>
            {uncertainty_items}
        </ul>
    </div>

    <div class="footer">
        <span>SpectraSense v1.0.0 &bull; Automated Evidence Layer</span>
        <span>Signal Intelligence &amp; Waveform Analysis</span>
    </div>
</div>
</body>
</html>
    """
    return html_content
