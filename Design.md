# Design.md — Analyst Dashboard UI/UX (Streamlit)

## Direction
Calm, dense-but-readable, "operations console". Personality: trustworthy, precise. Dark and light supported; risk colours carry meaning, never decoration.

## Colour System
| Role | Light | Dark |
|---|---|---|
| Primary | #2563EB | #60A5FA |
| Secondary | #475569 | #94A3B8 |
| Background | #F8FAFC | #0F172A |
| Surface | #FFFFFF | #1E293B |
| Text | #0F172A | #F1F5F9 |
| Muted | #64748B | #94A3B8 |
| Border | #E2E8F0 | #334155 |
| Success | #16A34A | #4ADE80 |
| Warning | #D97706 | #FBBF24 |
| Error / High risk | #DC2626 | #F87171 |
| Accent | #7C3AED | #A78BFA |
Risk bands: Low < 0.3 (success), Medium 0.3–0.7 (warning), High > 0.7 (error). Never rely on colour alone — add text labels. Maintain WCAG AA contrast (4.5:1).

## Typography
Inter (fallback system-ui). H1 28/600, H2 22/600, H3 18/600, body 14/400, labels 12/500 uppercase-optional, captions 12/400, monospace (JetBrains Mono) for IDs/amounts. Line height 1.5 body, 1.25 headings.

## Spacing
4-pt scale: 4, 8, 12, 16, 24, 32, 48.

## Layout
- **Sidebar:** navigation (Overview, Alerts, Case Detail, Model & Drift) + filters.
- **Overview:** KPI row (alerts today, % confirmed, avg score, p95 latency, drift status) → score distribution chart → recent alerts.
- **Alerts:** filterable table (status, risk band, date, amount), row click opens case.
- **Case detail:** transaction facts, score breakdown (XGB / GNN / blended), linked accounts (graph view), status control, label form.
- **Model & Drift:** current model version, PR-AUC, Evidently report embed, retrain button/history.
- Responsive: desktop-first (wide table); tablet stacks panels; mobile = single column, table → cards (best effort).

## Components
Buttons (primary/secondary/danger), select + multiselect filters, metric cards, data table with sortable columns, status badge (New / In review / Confirmed / False positive), risk badge, alert banner (drift), empty state ("No alerts match your filters"), loading spinner/skeleton, error state with retry.

## Interaction
Hover row highlight; visible focus ring (2px primary); status change shows toast confirmation; label form validates (required label) with inline message; auto-refresh toggle (default 10 s) for live alerts; transitions ≤ 150 ms; respect reduced motion.

## Accessibility
Keyboard-navigable tables and forms, labels on all inputs, text alternatives for charts (summary line), touch targets ≥ 44 px, non-colour risk cues.
