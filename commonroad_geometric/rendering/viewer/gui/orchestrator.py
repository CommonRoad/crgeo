from __future__ import annotations
from typing import TYPE_CHECKING, List
import pyglet
from pyglet import gl


if TYPE_CHECKING:
    import glooey


class GUIOrchestrator:
    """
    Experimental interface for managing interactive components
    integrated with a TrafficSceneRenderer. Uses glooey as backend: https://glooey.readthedocs.io/
    """

    def __init__(self, window: pyglet.window.Window) -> None:
        from glooey import Gui, Form
        from commonroad_geometric.common.utils.stack import magic_stack_reassignment
        from commonroad_geometric.rendering.viewer.gui.components import MagicReassignmentForm, MagicReassignmentFrame

        self._command_history: List[str] = []
        self._command_history_scroll_index: int = 0

        self.overlays = []

        self._gui = Gui(window)
    
        frame = MagicReassignmentFrame()
        form = MagicReassignmentForm("$")

        def on_unfocus(w: Form):
            command = w.get_text()
            self._command_history.append(command)
            self._command_history_scroll_index = len(self._command_history) - 1
            try:
                search_term, replace = command.split('=')
            except ValueError:
                w.set_text("$ >> please use the syntax 'foo=bar'")
                return
            num_reassignments = magic_stack_reassignment(
                search_term=search_term,
                replace=replace
            )
            w.set_text(f"$ >> {num_reassignments} reassignments (use mouse wheel to revisit cmd)")

        def on_focus(w: Form):
            if w.get_text()[0] == '$':
                w.set_text("")

        def on_mouse_scroll(x: int, y: int, dx: int, dy: int):
            if not self._command_history:
                return
            nonlocal form
            if dy > 0:
                self._command_history_scroll_index = max(0, self._command_history_scroll_index - 1)
            elif dy < 0:
                self._command_history_scroll_index = min(len(self._command_history) - 1, self._command_history_scroll_index + 1)
            form.set_text(self._command_history[self._command_history_scroll_index])

        form.push_handlers(
            on_unfocus=on_unfocus,
            on_focus=on_focus,
            on_mouse_scroll=on_mouse_scroll
        )
        frame.add(form)
        self._gui.add(frame)


    @property
    def gui(self) -> glooey.Gui:
        return self._gui

    def render_overlays(self) -> None:
        for o in self.overlays:
            #ty: Tuple[Tuple[float, float, float, float], List[Tuple[float, float]]]
            for ty in o.__get_objects__():
                
                gl.glColor4f(float(ty[0][0]), float(ty[0][1]), float(ty[0][2]), float(ty[0][3]))
                gl.glBegin(gl.GL_POLYGON)

                #gl.glVertex2f(0, 0)
                #gl.glVertex2f(self._viewer.window.get_size()[0], 0)
                #gl.glVertex2f(0, 10)
                for tt in ty[1]:
                    gl.glVertex2f(tt[0], tt[1])
                gl.glEnd()