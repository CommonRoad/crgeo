import pyglet
import glooey

class MagicReassignmentFrame(glooey.Frame):
    custom_alignment = 'bottom left'

    class Decoration(glooey.Background):
        custom_color = '#000000'

    class Box(glooey.Bin):
        custom_right_padding = 7
        custom_top_padding = 3
        custom_left_padding = 7
        custom_bottom_padding = 3

class MagicReassignmentForm(glooey.Form):
    class Label(glooey.EditableLabel):
        custom_font_name = 'Consolas'
        custom_font_size = 10
        custom_color = '#00ff00'
        custom_alignment = 'center'
        custom_width_hint = 600


class OverlayRadioButtonFrame(glooey.Frame):
    custom_alignment = 'top left'

    class Box(glooey.Bin):
        custom_right_padding = 2
        custom_top_padding = 2
        custom_left_padding = 2
        custom_bottom_padding = 2
    