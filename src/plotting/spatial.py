#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""(One liner introducing this file spatial.py)

(
Leading paragraphs explaining file in more detail.
)

Attributes:
    (Here, place any module-scope constants users will import.)
    
Constants:
    MODULE_LEVEL_CONSTANT1 (int): A module-level constant.

Examples:
    (Here, place useful implementations of the contents of spatial.py). Note that leading symbol '>>>' includes the 
    code in doctests, while '$' does not.)::
        
        >>> bar = 1
        >>> foo = bar + 1

(
Trailing paragraphs summarising final details.
)

Todo:
    * (Optional section for module-wide tasks).
    * (Use format: 'YYMMDD/task_identifier - one-liner task description'
    
References:
    Style guide: `Google Python Style Guide`_

Notes:
    File version
        0.1.0
    Project
        SpinChains-PythonAnalysis
    Path
        src/plotting/spatial.py
    Author
        Cameron Aidan McEleney < c.mceleney.1@research.gla.ac.uk >
    Created
        17 Jun 2025
    IDE
        PyCharm
        
.. _Google Python Style Guide:
   https://google.github.io/styleguide/pyguide.html
"""

# from __future__ import foo

# Standard library imports
from collections import namedtuple
from textwrap import dedent
from typing import Any, Optional

# Third-party imports
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle
import numpy as np

# Local application imports
# (e.g. from .helpers import foo

# Module-level constants
MODULE_LEVEL_CONSTANT1: int = 1
"""A module-level constant with in-line docstring."""

__all__ = ['ClickHandler', 'SpatialPlot']


class ClickHandler:

    def __init__(self):
        self.clicks = 0
        self.total = 0
        self.last_wavelength: Optional[float] = None

    def __call__(self, event: Any):

        # Right-click to reset
        if event.button == 3:
            self.clicks = self.total = 0
            self.last_wavelength = None
            print("Click data has been reset.")
            return

        self.clicks += 1
        x, y = event.xdata, event.ydata

        if self.last_wavelength is not None:
            diff = abs(x - self.last_wavelength)
            self.total += diff

            if self.clicks > 1:
                avg = self.total / (self.clicks - 1)
                print(f'Click #{self.clicks}: '
                      f'x: {event.xdata:.1f}, '
                      f'Avg. \u03BB: {avg:.1f}, '
                      f'Avg. k: {(2 * np.pi / avg):.3e} | '
                      f'y: {event.ydata:.3e}')
        else:
            print(f'Click #{self.clicks}: x: {event.xdata}, y: {event.ydata}')

        self.last_wavelength = x


class SpatialPlot:
    def __init__(self, owner):
        self._ = owner

    def _base_figure(
            self,
            index: int,
            is_single_figure: bool = True,
            inplace: bool = False,
            *,
            will_publish_plot: bool = False,
            should_highlight_regions: bool = False,
            **kwargs
    ) -> Figure:

        if self._.fig is None:
            fig = plt.figure(figsize=([4.4, 2.2]
                                      if is_single_figure
                                      else [4.4, 4.4]),
                             layout='constrained')
            self._.axes = fig.add_subplot(111)
            if not is_single_figure:
                # Adjusting for GIFs
                plt.rcParams.update({'savefig.dpi': 200, 'figure.dpi': 200})
        else:
            fig = self._.fig

        try:
            kwargs.get('axes')
        except AttributeError:
            pass
        else:
            if isinstance(kwargs['axes'], plt.Axes):
                self._.axes = kwargs.get('axes')
        finally:
            self._.axes.clear()
            self._.axes.set_aspect('auto')

        self._._yaxis_lim *= 8e-3 if is_single_figure else (max(self._.data_amplitude[index, :]))

        # Begin plotting
        self._.axes.plot(np.arange(0, self._.num_sites_total()),
                         self._.data_amplitude[index, :],
                         ls='-', lw='0.75', color='#64bb6a',
                         zorder=1.1,
                         label="Signal")

        _, y_major_labels, _ = self._._choose_scaling(subplot_to_scale=self._.axes)

        self._.axes.set(xlabel="Site index, n$_{i}$",
                        ylabel=f"m$_x$ (a.u. + {y_major_labels[1]} )",
                        xlim=[0.0, self._.num_sites_total()],
                        ylim=[-self._._yaxis_lim, self._._yaxis_lim])

        if will_publish_plot:
            self._.axes.text(x=-0.04,
                             y=0.96,
                             s=r'$\times \mathcal{10}^{{\mathcal{-3}}}$',
                             ha='center',
                             va='center',
                             transform=self._.axes.transAxes)

            self._.axes.text(x=0.88,
                             y=0.88,
                             s=f"(c) {self._.data_time[index]: 2.3f} ns",
                             fontsize=6,
                             va='center',
                             ha='center',
                             transform=self._.axes.transAxes)

        if should_highlight_regions:
            fields = ('lhs', 'driven', 'rhs')
            shape_position = namedtuple(
                typename='shape_positions',
                field_names=('lhs', 'driven', 'rhs'),
            )

            _bottom = self._.axes.get_ylim()[0] * 2
            anchors = shape_position(lhs=(0, _bottom),
                                     driven=(self._.driving_region_lhs() + self._.num_sites_abc(), _bottom),
                                     rhs=(self._.num_sites_total() - self._.num_sites_abc(), _bottom))

            widths = shape_position(lhs=self._.num_sites_abc(),
                                    driven=self._.driving_region_width(),
                                    rhs=self._.num_sites_abc())

            _height = 4 * self._.axes.get_ylim()[1]
            heights = shape_position(_height, _height, _height)

            for key in fields:
                rect = Rectangle(xy=getattr(anchors, key),
                                 width=getattr(widths, key),
                                 height=getattr(heights, key),
                                 lw=0,
                                 alpha=0.75 if key == 'rhs' else 0.5,
                                 facecolor='grey',
                                 edgecolor=None)

                plt.gca().add_patch(rect)

        fig.set_layout_engine(layout='constrained')

        if inplace:
            self._.fig = fig

        return fig

    def plot_time_instance(
            self,
            index: int = -1,
            *,
            should_annotate: bool = False,
            use_fixed_ylim: bool = False,
            **kwargs
    ) -> None:
        """"""
        self._._base_figure(index, inplace=True, should_highlight_regions=True)

        self._._set_ticks(self._.axes, 1000, 200, 3, 4, yaxis_num_decimals=1.1, show_sci_notation=False)

        _, y_major_labels, _ = self._._choose_scaling(value=abs(self._.axes.get_ylim()[1]))

        self._._cleanup_plot(self._.axes)

        if should_annotate:
            exchange_string = (
                f"Uniform Exc.: {self._.exchange_heisenberg_min()} (T)"
                if self._.exchange_heisenberg_min() == self._.exchange_heisenberg_max()
                else f"J$_{{min}}$ = {self._.exchange_heisenberg_min()} (T) |"
                     f"J$_{{max}}$ = {self._.exchange_heisenberg_min()} (T)"
            )

            params_txt_body = dedent(rf"""\
                H$_{{0}}$ = {self._.bias_zeeman_static()} (T)
                | N = {self._.num_sites_chain()}
                | $\alpha$ = {self._.gilbert_chain(): 2.2e}
                H$_{{D1}}$ = {self._.bias_zeeman_oscillating_1(): 2.2e} (T)
                | H$_{{D2}}$ = {self._.bias_zeeman_oscillating_2(): 2.2e} (T)
                {exchange_string}
                """)

            params_txt_bbox = dict(
                boxstyle='round',
                facecolor='gainsboro',
                alpha=0.5
            )

            self._.axes.text(x=0.05,
                             y=1.2,
                             s=params_txt_body,
                             fontsize=self._._font_sizes.small,
                             ha='center',
                             va='center',
                             bbox=params_txt_bbox,
                             transform=self._.axes.transAxes,)

            # Post processing
            self._.fig.savefig(f"{self._.output_path}_row{index}.png",
                             bbox_inches='tight')

            if self._.is_interactive:
                handler = ClickHandler()
                self._.fig.canvas.mpl_connect('button_press_event', handler)
                self._.fig.set_layout_engine('constrained')

            plt.close(self._.fig)



