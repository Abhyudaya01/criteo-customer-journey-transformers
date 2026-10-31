# Dataset statistics: criteo_w12h (12h windows, 62 windows)

## Journey length per user

| Per user | Mean | Median | P90 | P99 | Max |
|---|---|---|---|---|---|
| Impressions | 2.68 | 1 | 6 | 18 | 880 |
| Active windows | 2.25 | 1 | 5 | 13 | 60 |
| Journey span (first to last active window) | 12.20 | 1 | 42 | 59 | 62 |
| Distinct campaigns | 1.27 | 1 | 2 | 4 | 16 |

- Users: 6,142,256; active in one window only: 55.5%; one impression only: 51.7%; 5+ active windows: 10.4%
- Users with at least one conversion: 4.29%

## Sparsity

- Active share of user-windows within each journey span: 18.4%
- Active share of all user x window cells: 3.63%
- Prediction positions with no impression in the current window: 89.2%
- Positions with any campaign in the next window: 5.0%
- Conversion rate in next window: 0.373% if active now, 0.160% if not

## Conversion rate

- Overall (per position): 0.183%
- Train / val / test label windows: 0.186% / 0.169% / 0.188%
- Per label window: min 0.100% (window 20), max 0.488% (window 1); excluded as too small: none
- By slot of day: slot 0: 0.137%, slot 1: 0.230%

## Campaigns

- Top 10 / 50 / 100 campaigns' share of impressions: 18.6% / 45.2% / 60.8% (of 675)
- Next-window campaigns the user had never been shown before: 16.2%
- Positions (with a next-window campaign) that include a new-to-user campaign: 16.5%
