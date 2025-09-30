# TODO: refactor this mess

import pandas as pd
from dash import Dash, html, dash_table, Input, Output
import dash_bootstrap_components as dbc
from llm_adapters.adapter_factory import AdapterFactory

BOOL_TO_YESNO = {True: "Yes", False: "No"}

MONEY_COLS = ["Prompt $", "Completion $", "Request $"]
INT_COLS = ["Context", "Completion"]

def fetch_dataframe() -> pd.DataFrame:
    """Build a tidy DataFrame from llm_adapters."""
    def b(x): return bool(x)
    def na(x): return x if x is not None else None

    rows = []
    for m in AdapterFactory.get_supported_models():
        rows.append(
            {
                "Model": m.name,
                "Vendor": m.vendor_name,
                "Provider": m.provider_name,
                "Prompt $": m.cost.prompt,
                "Completion $": m.cost.completion,
                "Request $": m.cost.request,
                "Context": m.context_length,
                "Completion": na(m.completion_length),
                "User": b(m.can_user),
                "Repeating Roles": b(m.can_repeating_roles),
                "Streaming": b(m.supports_streaming),
                "Vision": b(m.supports_vision),
                "Tools": b(m.supports_tools),
                "Supports N": b(m.supports_n),
                "System": b(m.can_system),
                "Multiple Systems": b(m.can_system_multiple),
                "Empty Content": b(m.can_empty_content),
                "Tool Choice": b(m.supports_tools_choice),
                "Tool Choice Required": b(m.supports_tools_choice_required),
                "JSON Output": b(m.supports_json_output),
                "JSON Content": b(m.supports_json_content),
                "Last Assistant": b(m.can_assistant_last),
                "First Assistant": b(m.can_assistant_first),
                "Temperature": b(m.can_temperature),
                "Only System": b(m.can_system_only),
                "Only Assistant": b(m.can_assistant_only),
            }
        )
    return pd.DataFrame(rows)


def humanize_money(x) -> str:
    """Readable money string for tiny decimals."""
    if pd.isna(x):
        return ""
    n = float(x)
    if n == 0:
        return "$0"
    if n < 1e-6:
        return f"${n:.1e}"
    if n < 0.01:
        return f"${n:.2g}"
    return f"${n:,.6f}".rstrip("0").rstrip(".")


def make_display_df(df: pd.DataFrame) -> pd.DataFrame:
    """Create a display-friendly copy (strings), leaving raw df intact if needed later."""
    d = df.copy()
    for col in MONEY_COLS:
        d[col] = d[col].apply(humanize_money)
    for col in INT_COLS:
        d[col] = d[col].apply(lambda v: "" if pd.isna(v) else f"{int(v):,}")
    bool_cols = [c for c in d.columns if d[c].dtype == bool]
    d[bool_cols] = d[bool_cols].replace(BOOL_TO_YESNO)
    return d


DF_RAW = fetch_dataframe()
DF_DISPLAY = make_display_df(DF_RAW)

COLUMNS = [{"name": c, "id": c, "type": "any"} for c in DF_DISPLAY.columns]

def light_styles():
    return dict(
        style_header={
            "backgroundColor": "white",
            "color": "#475569",
            "fontWeight": "700",
            "borderBottom": "1px solid #e5e7eb",
        },
        style_cell={
            "backgroundColor": "white",
            "color": "#0f172a",
            "borderBottom": "1px solid #f1f5f9",
            "padding": "10px 12px",
            "minWidth": "90px",
            "fontSize": "14px",
            "whiteSpace": "nowrap",
        },
        style_data_conditional=[
            {"if": {"row_index": "even"}, "backgroundColor": "#f8fafc"},
            *[
                {
                    "if": {"column_id": col, "filter_query": f"{{{col}}} = Yes"},
                    "backgroundColor": "#e8f5e9",
                    "color": "#0f5132",
                }
                for col in DF_DISPLAY.columns
                if DF_DISPLAY[col].isin(["Yes", "No"]).any()
            ],
            *[
                {
                    "if": {"column_id": col, "filter_query": f"{{{col}}} = No"},
                    "backgroundColor": "#fde8e8",
                    "color": "#842029",
                }
                for col in DF_DISPLAY.columns
                if DF_DISPLAY[col].isin(["Yes", "No"]).any()
            ],
        ],
    )


def dark_styles():
    return dict(
        style_header={
            "backgroundColor": "#0f172a",
            "color": "#94a3b8",
            "fontWeight": "700",
            "borderBottom": "1px solid #1f2937",
        },
        style_cell={
            "backgroundColor": "#0b1220",
            "color": "#e5e7eb",
            "borderBottom": "1px solid #0f172a",
            "padding": "10px 12px",
            "minWidth": "90px",
            "fontSize": "14px",
            "whiteSpace": "nowrap",
        },
        style_data_conditional=[
            {"if": {"row_index": "even"}, "backgroundColor": "rgba(148,163,184,0.08)"},
            *[
                {
                    "if": {"column_id": col, "filter_query": f"{{{col}}} = Yes"},
                    "backgroundColor": "#11321d",
                    "color": "#b7f7cc",
                }
                for col in DF_DISPLAY.columns
                if DF_DISPLAY[col].isin(["Yes", "No"]).any()
            ],
            *[
                {
                    "if": {"column_id": col, "filter_query": f"{{{col}}} = No"},
                    "backgroundColor": "#2a1010",
                    "color": "#fbbbbb",
                }
                for col in DF_DISPLAY.columns
                if DF_DISPLAY[col].isin(["Yes", "No"]).any()
            ],
        ],
    )


TABLE_BASE = dict(
    fixed_rows={"headers": True},
    fixed_columns={"headers": True, "data": 1},
    filter_action="native",
    sort_action="native",
    page_action="none",
    style_table={"overflowX": "auto", "width": "100%", "minWidth": "100%"},
    style_cell_conditional=[
        {"if": {"column_id": "Model"}, "minWidth": "240px"},
        {"if": {"column_id": "Vendor"}, "minWidth": "140px"},
        {"if": {"column_id": "Provider"}, "minWidth": "140px"},
    ],
    css=[
        {"selector": "td:first-child, th:first-child",
         "rule": "position: sticky; left: 0; z-index: 10; background: inherit;"},
    ],
)

THEMES = {
    "Light": dbc.themes.BOOTSTRAP,
    "Dark": dbc.themes.DARKLY,
}

app = Dash(__name__, external_stylesheets=[THEMES["Light"]])
server = app.server

app.layout = dbc.Container(
    fluid=True,
    className="p-3",
    children=[
        dbc.Row(
            [
                dbc.Col(
                    [
                        html.H1("Supported Models", className="h3 mt-3 mb-1"),
                        html.P(
                            "Overview of vendor, provider, costs, and capabilities. "
                            "Use column filters; click headers to sort.",
                            className="text-muted",
                        ),
                    ],
                    md=8,
                ),
                dbc.Col(
                    [
                        dbc.RadioItems(
                            id="theme-picker",
                            options=[
                                {"label": "Light", "value": "Light"},
                                {"label": "Dark", "value": "Dark"},
                            ],
                            value="Light",
                            inline=True,
                            className="mt-3",
                            persistence=True,
                            persistence_type="local",
                        )
                    ],
                    md=4,
                    className="d-flex justify-content-md-end align-items-center",
                ),
            ]
        ),
        dbc.Row(
            dbc.Col(
                width=12,
                children=[
                    dash_table.DataTable(
                        id="models",
                        columns=COLUMNS,
                        data=DF_DISPLAY.to_dict("records"),
                        **TABLE_BASE,
                        **light_styles(),
                        tooltip_header={c["id"]: c["name"] for c in COLUMNS},
                        tooltip_delay=250,
                        tooltip_duration=None,
                    )
                ],
            ),
            className="mt-2",
        ),
        dbc.Row(
            dbc.Col(
                html.Small(
                    "Theme choice is saved locally and applied on reload.",
                    className="text-muted mt-2",
                )
            )
        ),
    ],
)


@app.callback(
    Output("models", "style_header"),
    Output("models", "style_cell"),
    Output("models", "style_data_conditional"),
    Output("models", "style_table"),
    Input("theme-picker", "value"),
    prevent_initial_call=False,
)
def apply_theme(theme):
    styles = dark_styles() if theme == "Dark" else light_styles()
    style_table = {"overflowX": "auto", "width": "100%", "minWidth": "100%"}
    return (
        styles["style_header"],
        styles["style_cell"],
        styles["style_data_conditional"],
        style_table,
    )


if __name__ == "__main__":
    app.run(debug=True)
