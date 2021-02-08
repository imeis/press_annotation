############################
## IMPORT
############################
from copy import deepcopy
import os, pickle, pytz
from datetime import datetime, timedelta
import numpy as np, pandas as pd
from functools import partial
import param, panel as pn, holoviews as hv
from holoviews.operation.datashader import datashade
from holoviews.streams import Stream
from bokeh.models import TickFormatter, FuncTickFormatter

pn.extension()
hv.extension('bokeh')

############################
## HELPERs
############################
pth_df = "df.csv"
id_names= [ '20171204081258',  '20171205065546',  '20171219065336',  '20171220064044',
            '20180219093754',  '20180220073721',  '20180222075958',  '20180306074102',
            '20180306113611',  '20180307065944',  '20180307120603',  '20180308064631',
            '20180308123124',  '20180309074335',  '20180327070011',  '20180327113846',
            '20180328055135',  '20180328110636',  '20180329060141',  '20180329130436',
            '20180403064139',  '20180403124319',  '20180404055340',  '20180404113333',
            '20180404115508',  '20180405054859',  '20180604070616',  '20180604112936',
            '20180604115111',  '20180605062753',  '20180605112153',  '20180606054636',
            '20180606102717',  '20180606120539',  '20180607053938',  '20180607122232',
            '20180607133002',  '20180608075408',  '20180626072116',  '20180626080915',
            '20180626121832',  '20180626161302',  '20180627053445',  '20180627120647',
            '20180628054648',  '20180628123927',  '20180629053618',  '20180629110053',
            '20180629110938',  '20180629112838',  '20180702064152',  '20180703065353',
            '20180703100844',  '20181008072057',  '20181009064425',  '20181010121306',
            '20181011054211',  '20181105084107',  '20181106072246',  '20181107071200',
            '20181108065911',  '20181108091743',  '20181109072122',  '20190603090338',
            '20190604060931',  '20190605054838',  '20190626054143',  '20190626161154',
            '20190627053601',  '20190627170241',  '20190628051852',  '20190628152111',
            '20190701052342',  '20190701154801',  '20190702051323',  '20190702095619',
            '20190703050641',  '20190703142247',  '20190704050858',  '20190704150045',
            '20190705054224',  '20190705152807',  '20190828052147',  '20190828142658',
            '20190829052043',  '20190829122758',  '20190830051556',  '20190830101007',
            '20190902071514',  '20190902095538',  '20190903052318',  '20190904051416',
            '20190904141858',  '20190905051243',  '20190905100229',  '20190906051720',
            '20190916053120',  '20190916111451',  '20190917051914',  '20190917121739',
            '20190918051556',  '20190918101130',  '20190919050811',  '20190919092626',
            '20190920051447',  '20190920091014',  '20190923051046',  '20190923095955',
            '20190924050601',  '20190924090134',  '20190925051836',  '20190926051544',
            '20190927050135',  '20190927082440',  '20190927084417',  '20191008052000',
            '20191009054104',  '20191009112038',  '20191010054317',  '20191010105126',
            '20191011051933',  '20191011113737',  '20191014052633',  '20191014100515',
            '20191015052038',  '20191016050531',  '20191016151130',  '20191017051603',
            '20191017142218',  '20191018051520',  '20191018102125',  ]

CLASSIFICATIONS = ["SR", "STC", "IOCA", "VF", "CPR", "SHK", "ROEA", "ASYS"]
#CLASSIFICATIONS = ["SinusRythmus", "SinusTachyCardia", "Induction", "VF", "CPR", "Shock", "ROEA", "Asystole"]

grid_style = {
    'grid_line_color': 'darkred',
    'grid_line_width': 1.5,
    'ygrid_bounds': (0, 1),
    'minor_xgrid_line_color': 'red',
    'xgrid_line_dash': [4, 4]
}
# This displays hour/minute marks for the x-ticks instead of milliseconds
def clock_ms_tick_formatter() -> TickFormatter:
    milliseconds_since_epoch = 0
    return FuncTickFormatter(
        code="""
        var d = new Date(initial + tick);
        return "" + d.getHours() + ":" + ("00" + d.getMinutes()).slice(-2) + ":" + ("00" + d.getSeconds()).slice(-2);
        """,
        args={"initial": milliseconds_since_epoch},
    )

opt = hv.opts(
    width=1200,
    height=250,
    xformatter=clock_ms_tick_formatter(),
    ylim=(-0.2,1.2),
    show_legend=False,
    gridstyle=grid_style,
    show_grid=True
)
df_default = pd.DataFrame(
    {},
    columns=[
        "mission_id",
        "annotation_time",
        "start_clock_ms",
        "end_clock_ms",
        "classification"
    ]
)
color_dict = {
       "SR": "blue",
       "STC": "lightgreen",
       "IOCA": "gray",
       "VF": "olive",
       "CPR": "cyan",
       "SHK": "goldenrod",
       "ROEA": "magenta",
       "ASYS": "red",
    }
COLORS = [
       "blue",
       "lightgreen",
       "gray",
       "olive",
       "cyan",
       "goldenrod",
       "magenta",
       "red",
]
with open("data/CPRs.pickle", 'rb') as f:
        cprs = pickle.load(f)

def normalize(x):
    mn = min(x)
    mx = max(x)
    if mx==mn:
        if mx==0: return x
        return x/mx
    return (x-mn)/(mx-mn)

def get_data(mission_id, norm=True):
    pth_processed = f"data\\processed_data_{mission_id}.pickle"
    with open(pth_processed, 'rb') as f:
        S = pickle.load(f)

    min_time = np.inf
    max_time = -np.inf

    def get_sensor(track, title, norm=norm):
        nonlocal min_time, max_time
        sensor = deepcopy(S[track])
        if sensor is None:
            title = title + "#NO DATA AVAILABLE FOR THIS PLOT#"
        if norm:
            sensor[:,1] = normalize(sensor[:,1])
        signals.append(sensor)
        titles.append(title)
        min_time = min([min_time, min(sensor[:,0])])
        max_time = max([max_time, max(sensor[:,0])])
    signals = []
    titles = []
    get_sensor("background_ecg", "Background ECG")
    get_sensor("p1", "Blood Pressure")
    get_sensor("spo2_curve", "Pulse")

    # CPR
    cpr = cprs[mission_id]
    if min(cpr[:,0])<min_time or min(cpr[:,0])>max_time:
        cpr = None
    return titles, signals, cpr


############################
## ANNOTATION CLASS
############################
class AnnotationMission(param.Parameterized):
    # (input) parameters
    cpr = param.Parameter()
    signals = param.Parameter()
    titles = param.Parameter()
    mission_id = param.Parameter()
    annotations = param.DataFrame(default=df_default)
    next_classification = param.ObjectSelector(default=CLASSIFICATIONS[0], objects=CLASSIFICATIONS)
    pending_start = param.Number(default=None)

    ############################
    ##  ANNOTATIONS PLOT
    ############################
    # Display the interactive elements, like annotations as colorful ranges over the plot
    def plot_annotations(self, **kwargs):
        flag = [str(i)==str(self.mission_id) for i in self.annotations.mission_id.values]
        rows = self.annotations[flag].iterrows()
        plots = []
        # We remember the first double-click and draw a vertical line if we expect another click to happen
        if self.pending_start:
            plots.append(
                hv.VLine(self.pending_start).opts(line_dash="dashed")
            )
        plots.extend([
            hv.VSpan(r["start_clock_ms"],r["end_clock_ms"]).opts(color=color_dict.get(r["classification"], "yellow"))#*
            #hv.Text((r["start_clock_ms"]+r["end_clock_ms"])/2,0.9,str(r["classification"])).opts(color="red")
            for ix, r in rows
        ])
        return hv.Overlay(plots)

    ############################
    ##  ANNOTATIONS REFRESH
    ############################
    def refresh_annotations(self):
        if hasattr(self, "_plot_update_stream"):
            self._plot_update_stream.event()

    ############################
    ##  SIGNALS PLOT
    ############################
    # Plot and datashade the ecg signal
    def plot_signal(self, **kwargs):

        curves = []
        curves.append(hv.Curve(self.cpr))
        curves.append(hv.Curve(self.signals[0]))
        curves.append(hv.Curve(self.signals[1], label=self.titles[1]).opts(opt))
        curves.append(hv.Curve(self.signals[2], label=self.titles[2]).opts(opt))

        return curves

    ############################
    ##  PLOT
    ############################

    def plot(self):
        signal_curves = self.plot_signal()
        # This is the clicking behaviour.
        self._plot_update_stream = hv.streams.Counter()

        def on_tap(x, y):
            # We have two modes, either there is no annotation pending,
            # so we remember the first click, or we record the annotation and reset the pending state.
            if not self.pending_start:
                self.pending_start = x
            else:
                values = (self.pending_start, x)
                start, end = min(values), max(values)
                self.annotations = self.annotations.append(pd.DataFrame({
                    "mission_id": [self.mission_id],
                    "annotation_time":[datetime.now()],
                    "start_clock_ms": [start],
                    "end_clock_ms": [end],
                    "classification": [self.next_classification],
                }), ignore_index=True)
                self.pending_start = None
            self.refresh_annotations()


        tap0 = hv.streams.DoubleTap(source=signal_curves[1])
        tap1 = hv.streams.DoubleTap(source=signal_curves[2])
        tap2 = hv.streams.DoubleTap(source=signal_curves[3])

        @tap0.add_subscriber
        def on_tap0(x,y):
            on_tap(x,y)

        @tap1.add_subscriber
        def on_tap1(x,y):
            on_tap(x,y)

        @tap2.add_subscriber
        def on_tap2(x,y):
            on_tap(x,y)


        ## annotation dynamic map
        annotations_dmap = hv.DynamicMap(
            self.plot_annotations,
            streams=[self._plot_update_stream]
        )

        ## ECG and CPR plot overlay
        ecg_opt = hv.opts.Overlay(title='ECG and CPR')
        ecg_curve = hv.Overlay([
            datashade(
                signal_curves[1],
                 cmap=["grey","black"]
            ).opts(opt),
            annotations_dmap,
        ])

        ecg_annot = hv.Overlay([
            ecg_curve,
            signal_curves[0].opts(color="red"),
        ]).opts(ecg_opt)

        ## output plot I
        output_plots = []
        output_plots.append(ecg_annot)

        ## output plot II
        output_plots.append(
            hv.Overlay([
                datashade(
                    signal_curves[2],
                    cmap=["grey", "black"]
                ).opts(opt),
                annotations_dmap,
                ]).opts(
                opt
            )
        )

        ## output plot III
        output_plots.append(
            hv.Overlay([
                datashade(
                    signal_curves[3],
                    cmap=["grey", "black"]
                ).opts(opt),
                annotations_dmap,
            ]).opts(
                opt
            )
        )
        return tuple(output_plots)

    ############################
    ##  ANNOTATION REMOVE
    ############################

    # These are the handlers for the "detail table"
    def on_remove_annotation(self, ix):
        self.annotations = self.annotations.drop(ix)
        self.refresh_annotations()

    ############################
    ##  ANNOTATION CHANGE
    ############################

    def on_change_annotation(self, ix, value):
        self.annotations.loc[ix, "classification"] = value
        # This line is needed to notify param of the inplace updated annotations dataframe
        self.annotations = self.annotations
        self.refresh_annotations()


    ############################
    ##  ANNOTATION SAVE
    ############################
    @param.depends("annotations")
    def action_save_annotations(self):
        try:
            self.annotations.to_csv(pth_df, mode='w')
        except:
            self.annotations.to_csv(pth_df)
    save_annotations = param.Action(action_save_annotations, doc="Save Changes", label="Save Mission Changes")
    ############################
    ##  CONTROL PANEL
    ############################

    # This is the detail table below where you can change the annotation made, or remove it.
    @param.depends("annotations")
    def plot_annotation_details(self):

        elements = []
        for i, (ix, r) in enumerate(
            self.annotations
            # Sorting the dataframe here is necessary,
            # otherwise we would number the ranges by their insertion, not by their time.
            .sort_values("start_clock_ms")
            .iterrows()
        ):
            if str(r["mission_id"])==str(self.mission_id):
                select = pn.widgets.RadioButtonGroup(
                    name="Select classification",
                    options=CLASSIFICATIONS,
                    value=r["classification"],
                    inline=True,
                )

                remove = pn.widgets.Button(
                    name="remove",
                    width=40,
                )
                clock_ms = int(float(r['start_clock_ms'])/1000)
                tstamp = datetime.fromtimestamp(clock_ms).strftime("%H:%M:%S")
                select.param.watch(partial(lambda ix, event: self.on_change_annotation(ix, event.new), ix), "value")
                remove.param.watch(partial(lambda ix, event: self.on_remove_annotation(ix), ix), "clicks")
                elements.extend(
                    [
                        pn.widgets.StaticText(name=f"@ {tstamp} ", value=""),
                        remove,
                        select

                    ]
                )
        return pn.GridBox(*elements, ncols=3, width=1200)

    def render(self):
        return pn.Column(

            pn.pane.Markdown('### Start annotating by double clicking into the plot. This will mark the start of a range. Double click again to mark the end of the range.', style={'font-family': "serif", 'color':"#ff0000"}),
            pn.Row(
                pn.pane.Markdown('### Classification for next annotation:', style={'font-family': "serif"}),
                pn.Param(self.param.next_classification, widgets={
                    "next_classification": pn.widgets.RadioButtonGroup(options=CLASSIFICATIONS,)#style={'font-size':'10pt'},css_classes=["widget-button"])
                }),
                pn.Spacer(background='white', width=100, height=10),
                #self.param.remove_last_annotation,
                self.param.save_annotations,
            ),
            *(self.plot()),
            pn.pane.Markdown(f"### List of annotations for mission {self.mission_id}", style={'font-family': "serif"}),
            self.plot_annotation_details,
        )

class SelectMission(param.Parameterized):
    mission_id = param.ObjectSelector(objects=[None]+id_names)
    app = param.Parameter()
    @param.depends("mission_id")
    def show_mission(self):
        if not self.mission_id:
            return None
        titles, signals,cpr = get_data(self.mission_id)
        try:
            annotations = pd.read_csv(pth_df, index_col=0)
        except FileNotFoundError:
            annotations = df_default

        app = AnnotationMission(
            cpr=cpr,
            signals=signals,
            titles = titles,
            mission_id=self.mission_id,
            annotations = annotations
        )
        self.app = app
        return app.render()

    def render(self):
        if self.app: self.annotations = self.app.annotations
        return pn.Column(
            pn.Param(self.param, parameters=["mission_id"]),
            self.show_mission,
        )
