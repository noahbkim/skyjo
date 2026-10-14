"""Presentation of fixed seats and renderer-independent diagnostics."""

from rich.console import Group
from rich.table import Table
from rich.text import Text

from skyjo.engine import game as sj
from skyjo.simulation.decisions import policy_section

PHASES = (
    "Reveal second card",
    "Draw or take",
    "Discard and flip, or replace",
    "Replace with taken card",
)


def card_text(card):
    if card == sj.FINGER_HIDDEN:
        return "?"
    if card == sj.FINGER_CLEARED:
        return "·"
    return str(card - 2)


def card_style(card):
    if card >= sj.FINGER_HIDDEN:
        return "dim"
    if card < sj.CARD_0:
        return "bold #4c8dff"
    if card == sj.CARD_0:
        return "bold #7dd3fc"
    if card <= sj.CARD_P4:
        return "bold #4ade80"
    if card <= sj.CARD_P8:
        return "bold #facc15"
    return "bold #f87171"


def current_card(state):
    phase = sj.get_action(state)
    label, instruction = (
        ("DRAWN CARD", "Replace a card, or discard and flip")
        if phase == sj.ACTION_FLIP_OR_REPLACE
        else ("TAKEN FROM DISCARD", "Choose a card to replace")
        if phase == sj.ACTION_REPLACE
        else (
            "DISCARD",
            "Reveal your second card first"
            if phase == sj.ACTION_FLIP_SECOND
            else "Available to take",
        )
    )
    card = sj.get_top(state)
    text = Text(f"{label}   ", style="bold", justify="center")
    text.append(
        "—" if card is None else f"[ {card_text(card)} ]",
        style=None if card is None else card_style(card),
    )
    text.append(
        f"\n{instruction if not sj.get_round_over(state) else 'Round complete'}",
        style="dim",
    )
    return text


def deck_summary(state):
    # The engine's unseen pool includes undealt values for hidden board cards.
    unseen = int(sj.get_draw_count(state))
    hidden = int(sj.get_table(state)[..., sj.FINGER_HIDDEN].sum())
    remaining = max(0, unseen - hidden)
    return Text.assemble(
        (f"DRAW PILE   {remaining} cards", "bold"),
        (f"\n{unseen} unseen · {hidden} on boards", "dim"),
        justify="center",
    )


def board(state, seat, name):
    current = sj.get_player(state)
    relative = (seat - current) % state.players
    total = int(sj.get_fixed_perspective_game_scores(state)[seat])
    title = Text(
        f"{'▶ ' if seat == current else ''}{seat + 1}. {name}",
        style="bold cyan" if seat == current else "bold",
    )
    table = Table(
        title=title,
        caption=f"Visible {sj.get_score(state, relative)} · Total {total}",
        expand=True,
        padding=(0, 1),
    )
    for column in range(sj.COLUMN_COUNT):
        table.add_column(str(column + 1), justify="center")
    for row in range(sj.ROW_COUNT):
        cells = []
        for column in range(sj.COLUMN_COUNT):
            card = sj.get_finger(state, row, column, relative)
            cells.append(Text(card_text(card), style=card_style(card)))
        table.add_row(*cells)
    return table


def diagnostics(frame, proposal=None):
    decision = proposal or frame.decision
    if decision is None:
        return Text(
            "No diagnostics at this position.\nStep back to inspect a completed decision."
        )
    sections = decision.diagnostics or (
        policy_section(
            frame.state, decision.probabilities, "Action selection probabilities"
        ),
    )
    output = []
    if proposal:
        output.append(Text("Proposed decision", style="bold cyan"))
    elif not any(section.row_actions for section in sections):
        output.append(
            Text(f"Chosen: {sj.get_action_name(frame.action)}", style="bold cyan")
        )
    for section in sections:
        output.append(Text(section.title, style="bold"))
        if section.text:
            output.append(Text(section.text))
        for key, value in section.metrics.items():
            output.append(Text(f"{key}: {format_scalar(value)}"))
        if section.columns:
            output.append(diagnostic_table(section, None if proposal else frame.action))
    return Group(*output)


def diagnostic_table(section, selected_action):
    table = Table()
    widths = []
    for index, heading in enumerate(section.columns):
        values = [row[index] for row in section.rows]
        numeric = bool(values) and all(
            value is None or value == "" or isinstance(value, (int, float))
            for value in values
        )
        width = max([len(heading), *(len(format_scalar(value)) for value in values)])
        if not numeric:
            width = min(width, 32)
        if index == 0 and section.row_actions:
            width += 2  # Reserve the selection marker without shifting row labels.
        widths.append(width)
        table.add_column(
            heading,
            width=width,
            no_wrap=numeric,
            justify="right" if numeric else "left",
        )
    # Preserve complete numbers; the containing pane scrolls horizontally.
    table.min_width = sum(widths) + 3 * len(widths) + 1
    for index, row in enumerate(section.rows):
        cells = [Text(format_scalar(value)) for value in row]
        style = None
        if section.row_actions:
            action = section.row_actions[index]
            selected = action is not None and action == selected_action
            cells[0] = Text("▶ " if selected else "  ") + cells[0]
            style = (
                "bold cyan reverse" if selected else "bold" if action is None else None
            )
        table.add_row(*cells, style=style)
    return table


def diagnostic_width(content):
    """Minimum viewport width needed to preserve every numeric table column."""
    if not isinstance(content, Group):
        return 0
    return max(
        (
            item.min_width or 0
            for item in content.renderables
            if isinstance(item, Table)
        ),
        default=0,
    )


def format_scalar(value):
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:.9g}"
    return str(value)
