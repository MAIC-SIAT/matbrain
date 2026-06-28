#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from collections import Counter
from datetime import datetime
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import yaml


ROOT = Path(__file__).resolve().parents[1]
STACK_PATH = ROOT / "mat_mcp_stack.yaml"


HTML = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Mat-MCP Load Dashboard</title>
  <style>
    :root {
      --bg: #0c1014;
      --panel: #131b22;
      --panel-2: #1a2530;
      --panel-3: #223241;
      --text: #e8edf2;
      --muted: #93a1b0;
      --ok: #4ade80;
      --warn: #f59e0b;
      --bad: #ef4444;
      --bad-soft: rgba(239, 68, 68, 0.14);
      --ok-soft: rgba(74, 222, 128, 0.14);
      --warn-soft: rgba(245, 158, 11, 0.14);
      --line: #2a3642;
      --accent: #7dd3fc;
      --accent-2: #38bdf8;
      --font: "IBM Plex Sans", "Source Sans 3", sans-serif;
      --mono: "IBM Plex Mono", "JetBrains Mono", monospace;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      font-family: var(--font);
      background:
        radial-gradient(circle at top left, rgba(56, 189, 248, 0.10), transparent 30%),
        radial-gradient(circle at top right, rgba(74, 222, 128, 0.08), transparent 20%),
        linear-gradient(180deg, #0c1014 0%, #0b1117 100%);
      color: var(--text);
    }
    .wrap {
      max-width: 1800px;
      margin: 0 auto;
      padding: 24px;
    }
    .topbar {
      display: grid;
      grid-template-columns: minmax(0, 1.5fr) minmax(360px, 1fr);
      gap: 16px;
      margin-bottom: 18px;
    }
    h1 {
      margin: 0;
      font-size: 30px;
      letter-spacing: -0.03em;
    }
    .sub {
      color: var(--muted);
      margin-top: 6px;
      font-size: 14px;
    }
    .controls {
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 12px;
      align-content: start;
    }
    .pill {
      border: 1px solid var(--line);
      border-radius: 999px;
      padding: 8px 12px;
      background: rgba(255,255,255,0.03);
      font-size: 13px;
      color: var(--muted);
    }
    select, button, input {
      background: var(--panel);
      color: var(--text);
      border: 1px solid var(--line);
      border-radius: 10px;
      padding: 8px 10px;
      font: inherit;
    }
    button { cursor: pointer; }
    .summary {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
      gap: 12px;
      margin-bottom: 16px;
    }
    .card {
      background: linear-gradient(180deg, rgba(255,255,255,0.02), rgba(255,255,255,0.01));
      border: 1px solid var(--line);
      border-radius: 16px;
      padding: 14px 16px;
    }
    .card .label {
      color: var(--muted);
      font-size: 12px;
      text-transform: uppercase;
      letter-spacing: 0.08em;
    }
    .card .value {
      margin-top: 8px;
      font-size: 28px;
      font-weight: 700;
    }
    .layout {
      display: grid;
      grid-template-columns: 390px minmax(0, 1fr);
      gap: 16px;
      align-items: start;
    }
    .sidebar {
      display: grid;
      gap: 16px;
      position: sticky;
      top: 16px;
    }
    .panel {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 18px;
      overflow: hidden;
    }
    .panel-head {
      padding: 14px 16px;
      border-bottom: 1px solid var(--line);
      background: var(--panel-2);
      display: flex;
      justify-content: space-between;
      align-items: center;
      gap: 12px;
    }
    .panel-title {
      font-size: 16px;
      font-weight: 700;
    }
    .panel-body {
      padding: 14px 16px;
    }
    .filter-grid {
      display: grid;
      grid-template-columns: 1fr;
      gap: 10px;
    }
    .filter-grid label {
      display: grid;
      gap: 6px;
      font-size: 13px;
      color: var(--muted);
    }
    .hotspot-list {
      display: grid;
      gap: 10px;
    }
    .hotspot {
      display: grid;
      gap: 6px;
      border: 1px solid rgba(255,255,255,0.07);
      border-radius: 14px;
      background: rgba(255,255,255,0.02);
      padding: 12px;
    }
    .hotspot-top {
      display: flex;
      justify-content: space-between;
      gap: 10px;
      align-items: baseline;
    }
    .hotspot-name {
      font-weight: 700;
    }
    .hotspot-metric {
      color: var(--bad);
      font-weight: 700;
    }
    .tiny {
      font-size: 12px;
      color: var(--muted);
    }
    .service-grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(520px, 1fr));
      gap: 16px;
    }
    .service {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 18px;
      overflow: hidden;
    }
    .service-header {
      padding: 16px;
      border-bottom: 1px solid var(--line);
      background: var(--panel-2);
      display: flex;
      justify-content: space-between;
      gap: 12px;
      align-items: center;
      flex-wrap: wrap;
    }
    .service-title {
      font-size: 18px;
      font-weight: 700;
    }
    .service-meta {
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      margin-top: 10px;
    }
    .mini-pill {
      display: inline-flex;
      gap: 6px;
      align-items: center;
      padding: 5px 9px;
      border-radius: 999px;
      border: 1px solid rgba(255,255,255,0.08);
      background: rgba(255,255,255,0.03);
      color: var(--muted);
      font-size: 12px;
    }
    .status-badge {
      display: inline-flex;
      align-items: center;
      gap: 8px;
      padding: 8px 10px;
      border-radius: 999px;
      font-size: 13px;
      font-weight: 700;
      border: 1px solid rgba(255,255,255,0.08);
    }
    .status-badge.ok { color: var(--ok); background: var(--ok-soft); }
    .status-badge.warn { color: var(--warn); background: var(--warn-soft); }
    .status-badge.bad { color: var(--bad); background: var(--bad-soft); }
    .status.ok { color: var(--ok); }
    .status.warn { color: var(--warn); }
    .status.bad { color: var(--bad); }
    .service-body {
      padding: 14px 16px 16px;
      display: grid;
      gap: 14px;
    }
    .metric-row {
      display: grid;
      grid-template-columns: repeat(4, minmax(0, 1fr));
      gap: 10px;
    }
    .metric-row.compact {
      grid-template-columns: repeat(6, minmax(0, 1fr));
    }
    .metric-box {
      border: 1px solid rgba(255,255,255,0.07);
      border-radius: 14px;
      padding: 10px 12px;
      background: rgba(255,255,255,0.02);
    }
    .metric-box .m-label {
      color: var(--muted);
      font-size: 11px;
      text-transform: uppercase;
      letter-spacing: 0.07em;
    }
    .metric-box .m-value {
      margin-top: 6px;
      font-size: 22px;
      font-weight: 700;
    }
    .backend-grid {
      display: grid;
      gap: 10px;
    }
    .backend-card {
      border: 1px solid rgba(255,255,255,0.07);
      border-radius: 14px;
      padding: 12px;
      background: rgba(255,255,255,0.02);
    }
    .backend-head {
      display: flex;
      justify-content: space-between;
      gap: 12px;
      align-items: center;
      margin-bottom: 10px;
    }
    .backend-name {
      font-family: var(--mono);
      font-size: 12px;
      word-break: break-all;
    }
    .bar-group {
      display: grid;
      gap: 8px;
    }
    .bar-row {
      display: grid;
      grid-template-columns: 88px 1fr 52px;
      gap: 10px;
      align-items: center;
      font-size: 12px;
      color: var(--muted);
    }
    .bar-track {
      width: 100%;
      height: 9px;
      background: rgba(255,255,255,0.06);
      border-radius: 999px;
      overflow: hidden;
    }
    .bar-fill {
      height: 100%;
      border-radius: 999px;
      background: linear-gradient(90deg, var(--accent), var(--accent-2));
    }
    .bar-fill.warn {
      background: linear-gradient(90deg, #fbbf24, #f59e0b);
    }
    .bar-fill.bad {
      background: linear-gradient(90deg, #fb7185, #ef4444);
    }
    table {
      width: 100%;
      border-collapse: collapse;
      font-size: 13px;
    }
    th, td {
      text-align: left;
      padding: 10px 12px;
      border-bottom: 1px solid rgba(255,255,255,0.06);
      vertical-align: top;
    }
    th {
      color: var(--muted);
      font-weight: 600;
      background: rgba(255,255,255,0.02);
    }
    td.mono, .mono {
      font-family: var(--mono);
      font-size: 12px;
    }
    .events {
      max-height: 340px;
      overflow: auto;
      padding-right: 4px;
    }
    .detail-sections {
      display: grid;
      gap: 10px;
    }
    .detail-toggle {
      border: 1px solid rgba(255,255,255,0.07);
      border-radius: 14px;
      background: rgba(255,255,255,0.02);
      overflow: hidden;
    }
    .detail-toggle summary {
      cursor: pointer;
      list-style: none;
      padding: 12px 14px;
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 12px;
      font-size: 13px;
      font-weight: 600;
    }
    .detail-toggle summary::-webkit-details-marker {
      display: none;
    }
    .detail-toggle summary::after {
      content: '+';
      color: var(--muted);
      font-size: 16px;
      font-weight: 700;
    }
    .detail-toggle[open] summary::after {
      content: '-';
    }
    .detail-body {
      padding: 0 14px 14px;
    }
    .tiny-inline {
      color: var(--muted);
      font-size: 12px;
      font-weight: 400;
    }
    .event {
      padding: 10px 12px;
      border: 1px solid rgba(255,255,255,0.06);
      border-radius: 12px;
      margin-bottom: 10px;
      background: rgba(255,255,255,0.02);
    }
    .event .k {
      color: var(--muted);
    }
    .err {
      color: var(--bad);
      white-space: pre-wrap;
      word-break: break-word;
    }
    .muted { color: var(--muted); }
    .hidden { display: none !important; }
    .empty-state {
      border: 1px dashed rgba(255,255,255,0.12);
      border-radius: 16px;
      padding: 24px;
      text-align: center;
      color: var(--muted);
      background: rgba(255,255,255,0.02);
    }
    @media (max-width: 1280px) {
      .layout { grid-template-columns: 1fr; }
      .sidebar { position: static; }
    }
    @media (max-width: 860px) {
      .topbar { grid-template-columns: 1fr; }
      .controls { grid-template-columns: 1fr; }
      .metric-row { grid-template-columns: repeat(2, minmax(0, 1fr)); }
      .metric-row.compact { grid-template-columns: repeat(2, minmax(0, 1fr)); }
    }
  </style>
</head>
<body>
  <div class="wrap">
    <div class="topbar">
      <div>
        <h1>Mat-MCP Load Dashboard</h1>
        <div class="sub">Router-level live load, backend fanout, and recent request failures.</div>
      </div>
      <div class="controls">
        <label class="pill">Auto refresh
          <select id="interval">
            <option value="2">2s</option>
            <option value="5" selected>5s</option>
            <option value="10">10s</option>
            <option value="0">off</option>
          </select>
        </label>
        <button id="refresh">Refresh now</button>
        <span class="pill" id="stamp">never</span>
        <span class="pill" id="server-meta">booting</span>
      </div>
    </div>

    <div class="summary" id="summary"></div>
    <div class="layout">
      <aside class="sidebar">
        <section class="panel">
          <div class="panel-head">
            <div class="panel-title">Filters</div>
          </div>
          <div class="panel-body filter-grid">
            <label>
              Search service / tool
              <input id="search" placeholder="matgl, uip, band_gap, validate..." />
            </label>
            <label>
              Status
              <select id="status-filter">
                <option value="all">all</option>
                <option value="bad">bad</option>
                <option value="warn">warn</option>
                <option value="ok">ok</option>
              </select>
            </label>
            <label>
              Sort by
              <select id="sort-by">
                <option value="failures">failures</option>
                <option value="inflight">inflight</option>
                <option value="sessions">sessions</option>
                <option value="activity">activity</option>
                <option value="name">name</option>
              </select>
            </label>
          </div>
        </section>
        <section class="panel">
          <div class="panel-head">
            <div class="panel-title">Failure Hotspots</div>
            <div class="tiny">top services by failed connects/posts</div>
          </div>
          <div class="panel-body hotspot-list" id="service-hotspots"></div>
        </section>
        <section class="panel">
          <div class="panel-head">
            <div class="panel-title">Tool Hotspots</div>
            <div class="tiny">recent failing tools across routers</div>
          </div>
          <div class="panel-body hotspot-list" id="tool-hotspots"></div>
        </section>
      </aside>
      <main>
        <div class="service-grid" id="services"></div>
      </main>
    </div>
  </div>

  <script>
    let timer = null;
    let latestData = null;

    function setTimer() {
      if (timer) clearInterval(timer);
      const value = Number(document.getElementById('interval').value);
      if (value > 0) timer = setInterval(loadData, value * 1000);
    }

    function metricCard(label, value) {
      return `<div class="card"><div class="label">${label}</div><div class="value">${value}</div></div>`;
    }

    function serviceStatus(service) {
      if (!service.ok) return ['bad', 'unreachable'];
      const agg = service.aggregates;
      const failed = agg.failed_connects + agg.failed_posts;
      const inflight = agg.inflight_post + agg.active_sse;
      if (failed > 0) return ['warn', 'degraded'];
      if (inflight > 0) return ['ok', 'active'];
      return ['ok', 'idle'];
    }

    function ratioFill(numerator, denominator, defaultRatio = 0) {
      if (!denominator) return `${Math.round(defaultRatio * 100)}%`;
      return `${Math.min(100, Math.round((numerator / denominator) * 100))}%`;
    }

    function renderSummary(data) {
      const totalServices = data.summary.total_services;
      const reachable = data.summary.reachable_services;
      const totalSessions = data.summary.sessions_tracked;
      const totalInflight = data.summary.inflight_post;
      const totalActiveSse = data.summary.active_sse;
      const totalFailedPosts = data.summary.failed_posts;
      const totalFailedConnects = data.summary.failed_connects;
      document.getElementById('summary').innerHTML = [
        metricCard('Services reachable', `${reachable}/${totalServices}`),
        metricCard('Tracked sessions', totalSessions),
        metricCard('Active SSE', totalActiveSse),
        metricCard('Inflight posts', totalInflight),
        metricCard('Failed posts', totalFailedPosts),
        metricCard('Failed connects', totalFailedConnects),
      ].join('');
      document.getElementById('server-meta').textContent = `${data.host} | ${totalServices} services`;
    }

    function aggregateBackends(service) {
      const backends = service.payload?.backends || [];
      if (!backends.length) {
        return '<div class="muted">No backend rows.</div>';
      }
      const maxActivity = Math.max(1, ...backends.map(row => row.active_sse + row.inflight_post + row.tracked_sessions));
      const maxFailures = Math.max(1, ...backends.map(row => row.failed_connects + row.failed_posts));
      return backends.map(row => {
        const activity = row.active_sse + row.inflight_post + row.tracked_sessions;
        const failures = row.failed_connects + row.failed_posts;
        const activityClass = activity > 0 ? (activity >= maxActivity * 0.7 ? 'bad' : activity >= maxActivity * 0.4 ? 'warn' : '') : '';
        const failureClass = failures > 0 ? (failures >= maxFailures * 0.7 ? 'bad' : 'warn') : '';
        return `
          <div class="backend-card">
            <div class="backend-head">
              <div class="backend-name">${row.backend}</div>
              <div class="mini-pill">POST ${row.total_post} | SSE ${row.total_sse}</div>
            </div>
            <div class="bar-group">
              <div class="bar-row">
                <div>active SSE</div>
                <div class="bar-track"><div class="bar-fill ${activityClass}" style="width:${ratioFill(row.active_sse, maxActivity)}"></div></div>
                <div>${row.active_sse}</div>
              </div>
              <div class="bar-row">
                <div>sessions</div>
                <div class="bar-track"><div class="bar-fill ${activityClass}" style="width:${ratioFill(row.tracked_sessions, maxActivity)}"></div></div>
                <div>${row.tracked_sessions}</div>
              </div>
              <div class="bar-row">
                <div>inflight</div>
                <div class="bar-track"><div class="bar-fill ${activityClass}" style="width:${ratioFill(row.inflight_post, maxActivity)}"></div></div>
                <div>${row.inflight_post}</div>
              </div>
              <div class="bar-row">
                <div>failures</div>
                <div class="bar-track"><div class="bar-fill ${failureClass}" style="width:${ratioFill(row.failed_connects + row.failed_posts, maxFailures)}"></div></div>
                <div>${row.failed_connects + row.failed_posts}</div>
              </div>
            </div>
          </div>
        `;
      }).join('');
    }

    function renderHotspots(data) {
      const serviceHotspots = (data.hotspots.services || []).slice(0, 8).map(item => `
        <div class="hotspot">
          <div class="hotspot-top">
            <div class="hotspot-name">${item.name}</div>
            <div class="hotspot-metric">${item.failures}</div>
          </div>
          <div class="tiny">failed posts ${item.failed_posts} | failed connects ${item.failed_connects}</div>
          <div class="tiny">inflight ${item.inflight_post} | sessions ${item.sessions_tracked}</div>
        </div>
      `).join('') || '<div class="empty-state">No service failures yet.</div>';
      const toolHotspots = (data.hotspots.tools || []).slice(0, 8).map(item => `
        <div class="hotspot">
          <div class="hotspot-top">
            <div class="hotspot-name">${item.tool_name}</div>
            <div class="hotspot-metric">${item.count}</div>
          </div>
          <div class="tiny">${item.service_count} services | last seen in recent failures</div>
        </div>
      `).join('') || '<div class="empty-state">No failing tool events yet.</div>';
      document.getElementById('service-hotspots').innerHTML = serviceHotspots;
      document.getElementById('tool-hotspots').innerHTML = toolHotspots;
    }

    function filteredServices(data) {
      const query = document.getElementById('search').value.trim().toLowerCase();
      const statusFilter = document.getElementById('status-filter').value;
      const sortBy = document.getElementById('sort-by').value;
      const rows = data.services.slice().filter(service => {
        const [klass] = serviceStatus(service);
        if (statusFilter !== 'all' && klass !== statusFilter) return false;
        if (!query) return true;
        const hay = [
          service.name,
          service.load_url,
          ...(service.payload?.recent_events || []).map(evt => `${evt.tool_name || ''} ${evt.rpc_method || ''} ${evt.error || ''}`),
        ].join(' ').toLowerCase();
        return hay.includes(query);
      });
      rows.sort((a, b) => {
        const aa = a.aggregates || {};
        const bb = b.aggregates || {};
        if (sortBy === 'name') return a.name.localeCompare(b.name);
        if (sortBy === 'sessions') return (bb.sessions_tracked || 0) - (aa.sessions_tracked || 0);
        if (sortBy === 'inflight') return (bb.inflight_post || 0) - (aa.inflight_post || 0);
        if (sortBy === 'activity') return ((bb.inflight_post || 0) + (bb.active_sse || 0)) - ((aa.inflight_post || 0) + (aa.active_sse || 0));
        return ((bb.failed_posts || 0) + (bb.failed_connects || 0)) - ((aa.failed_posts || 0) + (aa.failed_connects || 0));
      });
      return rows;
    }

    function renderServices(data) {
      const services = filteredServices(data);
      const root = document.getElementById('services');
      if (!services.length) {
        root.innerHTML = '<div class="empty-state">No services match the current filter.</div>';
        return;
      }
      root.innerHTML = services.map(service => {
        const [klass, statusText] = serviceStatus(service);
        if (!service.ok) {
          return `
            <section class="service">
              <div class="service-header">
                <div>
                  <div class="service-title">${service.name}</div>
                  <div class="muted mono">${service.load_url}</div>
                </div>
                <div class="status-badge ${klass}">${statusText}</div>
              </div>
              <div class="service-body">
                <div class="empty-state err">${service.error || 'unreachable'}</div>
              </div>
            </section>`;
        }
        const agg = service.aggregates;
        const allEvents = (service.payload.recent_events || []).slice().reverse();
        const shownEvents = allEvents.slice(0, 6);
        const events = shownEvents.map(evt => `
          <div class="event mono">
            <div><span class="k">kind</span>: ${evt.kind || ''}</div>
            <div><span class="k">backend</span>: ${evt.backend || ''}</div>
            ${evt.session_id ? `<div><span class="k">session</span>: ${evt.session_id}</div>` : ''}
            ${evt.rpc_method ? `<div><span class="k">rpc</span>: ${evt.rpc_method}</div>` : ''}
            ${evt.tool_name ? `<div><span class="k">tool</span>: ${evt.tool_name}</div>` : ''}
            ${evt.error ? `<div class="err"><span class="k">error</span>: ${evt.error}</div>` : ''}
          </div>
        `).join('') || '<div class="muted">No recent events.</div>';
        const detailSections = `
          <div class="detail-sections">
            <details class="detail-toggle">
              <summary>
                <span>Backends</span>
                <span class="tiny-inline">${agg.backend_count} backend${agg.backend_count === 1 ? '' : 's'}</span>
              </summary>
              <div class="detail-body">
                <div class="backend-grid">${aggregateBackends(service)}</div>
              </div>
            </details>
            <details class="detail-toggle">
              <summary>
                <span>Recent events</span>
                <span class="tiny-inline">showing ${Math.min(allEvents.length, 6)} / ${allEvents.length}</span>
              </summary>
              <div class="detail-body">
                <div class="events">${events}</div>
              </div>
            </details>
          </div>
        `;
        return `
          <section class="service">
            <div class="service-header">
              <div>
                <div class="service-title">${service.name}</div>
                <div class="muted mono">${service.load_url}</div>
                <div class="service-meta">
                  <span class="mini-pill">backends ${agg.backend_count}</span>
                  <span class="mini-pill">sessions ${agg.sessions_tracked}</span>
                  <span class="mini-pill">active SSE ${agg.active_sse}</span>
                  <span class="mini-pill">recent failures ${agg.recent_failures}</span>
                </div>
              </div>
              <div class="status-badge ${klass}">${statusText}</div>
            </div>
            <div class="service-body">
              <div class="metric-row compact">
                <div class="metric-box"><div class="m-label">Backends</div><div class="m-value">${agg.backend_count}</div></div>
                <div class="metric-box"><div class="m-label">Sessions</div><div class="m-value">${agg.sessions_tracked}</div></div>
                <div class="metric-box"><div class="m-label">Active SSE</div><div class="m-value">${agg.active_sse}</div></div>
                <div class="metric-box"><div class="m-label">Inflight POST</div><div class="m-value">${agg.inflight_post}</div></div>
                <div class="metric-box"><div class="m-label">Failed POST</div><div class="m-value">${agg.failed_posts}</div></div>
                <div class="metric-box"><div class="m-label">Failed Connect</div><div class="m-value">${agg.failed_connects}</div></div>
              </div>
              ${detailSections}
            </div>
          </section>`;
      }).join('');
    }

    async function loadData() {
      const resp = await fetch('/api/load');
      const data = await resp.json();
      latestData = data;
      renderSummary(data);
      renderHotspots(data);
      renderServices(data);
      const stamp = data.generated_at || new Date().toISOString();
      document.getElementById('stamp').textContent = new Date(stamp).toLocaleTimeString();
    }

    document.getElementById('refresh').addEventListener('click', loadData);
    document.getElementById('interval').addEventListener('change', setTimer);
    document.getElementById('search').addEventListener('input', () => latestData && renderServices(latestData));
    document.getElementById('status-filter').addEventListener('change', () => latestData && renderServices(latestData));
    document.getElementById('sort-by').addEventListener('change', () => latestData && renderServices(latestData));
    setTimer();
    loadData();
  </script>
</body>
</html>
"""


def enabled_services(stack_path: Path) -> list[dict[str, Any]]:
    data = yaml.safe_load(stack_path.read_text(encoding="utf-8")) or {}
    return [item for item in data.get("services", []) if item.get("enabled", True)]


def fetch_json(url: str, timeout: float) -> tuple[bool, dict[str, Any] | None, str | None]:
    try:
        req = Request(url, headers={"Accept": "application/json"})
        with urlopen(req, timeout=timeout) as resp:  # noqa: S310
            payload = json.loads(resp.read().decode("utf-8"))
        return True, payload, None
    except HTTPError as exc:
        return False, None, f"HTTP {exc.code}: {exc.reason}"
    except URLError as exc:
        return False, None, repr(exc.reason)
    except Exception as exc:  # noqa: BLE001
        return False, None, repr(exc)


def build_aggregates(payload: dict[str, Any] | None) -> dict[str, Any]:
    payload = payload or {}
    backends = payload.get("backends") or []
    recent_events = payload.get("recent_events") or []
    recent_failures = [evt for evt in recent_events if evt.get("kind") in {"post_failed", "backend_connect_failed"}]
    return {
        "backend_count": len(backends),
        "sessions_tracked": int(payload.get("sessions_tracked") or 0),
        "active_sse": sum(int(row.get("active_sse") or 0) for row in backends),
        "inflight_post": sum(int(row.get("inflight_post") or 0) for row in backends),
        "total_post": sum(int(row.get("total_post") or 0) for row in backends),
        "failed_connects": sum(int(row.get("failed_connects") or 0) for row in backends),
        "failed_posts": sum(int(row.get("failed_posts") or 0) for row in backends),
        "recent_failures": len(recent_failures),
    }


def build_summary(services: list[dict[str, Any]]) -> dict[str, Any]:
    reachable = [service for service in services if service.get("ok")]
    return {
        "total_services": len(services),
        "reachable_services": len(reachable),
        "sessions_tracked": sum(int(service["aggregates"]["sessions_tracked"]) for service in services),
        "active_sse": sum(int(service["aggregates"]["active_sse"]) for service in services),
        "inflight_post": sum(int(service["aggregates"]["inflight_post"]) for service in services),
        "failed_posts": sum(int(service["aggregates"]["failed_posts"]) for service in services),
        "failed_connects": sum(int(service["aggregates"]["failed_connects"]) for service in services),
    }


def build_hotspots(services: list[dict[str, Any]]) -> dict[str, Any]:
    service_rows = []
    tool_counter: Counter[str] = Counter()
    tool_services: dict[str, set[str]] = {}
    for service in services:
        agg = service["aggregates"]
        failures = int(agg["failed_posts"]) + int(agg["failed_connects"])
        service_rows.append(
            {
                "name": service["name"],
                "failures": failures,
                "failed_posts": int(agg["failed_posts"]),
                "failed_connects": int(agg["failed_connects"]),
                "inflight_post": int(agg["inflight_post"]),
                "sessions_tracked": int(agg["sessions_tracked"]),
            }
        )
        for evt in (service.get("payload") or {}).get("recent_events") or []:
            if evt.get("kind") != "post_failed":
                continue
            tool_name = evt.get("tool_name")
            if not tool_name:
                continue
            tool_name = str(tool_name)
            tool_counter[tool_name] += 1
            tool_services.setdefault(tool_name, set()).add(service["name"])
    service_rows.sort(key=lambda row: (-row["failures"], -row["inflight_post"], row["name"]))
    tool_rows = [
        {
            "tool_name": tool_name,
            "count": count,
            "service_count": len(tool_services.get(tool_name, set())),
        }
        for tool_name, count in tool_counter.most_common(12)
    ]
    return {"services": service_rows[:12], "tools": tool_rows}


def collect_snapshot(stack_path: Path, host: str, timeout: float) -> dict[str, Any]:
    services = []
    for item in enabled_services(stack_path):
        port = int(item["port"])
        load_url = f"http://{host}:{port}/load"
        ok, payload, error = fetch_json(load_url, timeout)
        aggregates = build_aggregates(payload)
        services.append(
            {
                "name": item["name"],
                "port": port,
                "load_url": load_url,
                "ok": ok,
                "payload": payload,
                "error": error,
                "aggregates": aggregates,
            }
        )
    return {
        "host": host,
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "service_count": len(services),
        "summary": build_summary(services),
        "hotspots": build_hotspots(services),
        "services": services,
    }


class Handler(BaseHTTPRequestHandler):
    stack_path: Path = STACK_PATH
    host_value: str = "localhost"
    timeout_value: float = 2.0

    def do_GET(self) -> None:  # noqa: N802
        if self.path in {"/", "/index.html"}:
            body = HTML.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        if self.path == "/api/load":
            payload = collect_snapshot(self.stack_path, self.host_value, self.timeout_value)
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        self.send_response(404)
        self.end_headers()

    def log_message(self, format: str, *args: object) -> None:  # noqa: A003
        return


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Serve a local dashboard for Mat-MCP router /load endpoints.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8091)
    parser.add_argument("--mcp-host", default="localhost")
    parser.add_argument("--stack", type=Path, default=STACK_PATH)
    parser.add_argument("--timeout", type=float, default=2.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    Handler.stack_path = args.stack
    Handler.host_value = args.mcp_host
    Handler.timeout_value = args.timeout
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(
        json.dumps(
            {
                "service": "mcp-load-dashboard",
                "url": f"http://{args.host}:{args.port}/",
                "api": f"http://{args.host}:{args.port}/api/load",
                "mcp_host": args.mcp_host,
                "stack": str(args.stack),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    server.serve_forever()


if __name__ == "__main__":
    main()
