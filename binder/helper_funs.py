import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
import matplotlib.pyplot as plt
import gpflow
import xarray as xr
import plotly.express as px
import plotly.graph_objects as go
from shgp import SHGP
import plotly.io as pio
pio.renderers.default = 'jupyterlab'
import plotly
from tqdm import trange

# Hide warnings
import warnings
warnings.filterwarnings(action='ignore')

# Seeding for reproducibility
SEED = 30
rng = np.random.RandomState(SEED)
tfp_seed = tfp.random.sanitize_seed(SEED)

# Plotly specifics
import plotly.io as pio
import plotly.graph_objects as go
pio.templates["pres"] = go.layout.Template(
    layout=go.Layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        colorway=px.colors.qualitative.Set2,
        legend=dict(itemsizing='trace', font_size=22),
        font_size=22,
    )
)

config= dict(displayModeBar=False)

# Set plotly defaults
pio.templates.default = 'none+pres'
full_fig_width = 1000
full_fig_height = 600
half_fig_width = full_fig_width // 2
half_fig_height = full_fig_height // 2

# Plotting functionality
def plot_clim(da):
    
    da = da[:5] - 273.15
    da = da.sel(time=slice('1960', '2014'))
    fig1 = go.Figure()
    for i in range(len(da.realisation)):
        fig1.add_trace(
            go.Scatter(
                x=da.time,
                y=da[i],
                name=f'Model {i + 1}',
                line=dict(width=2)))
        
    fig1.add_trace(
            go.Scatter(
                x=da.time,
                y=da.mean('realisation'),
                name="'Average'",
                visible="legendonly",
                line=dict(color="black", width=4)))
    fig1.update_layout(
        yaxis_title='Global mean surface temp. (°C)',
        width=full_fig_width,
        height=full_fig_height)
    
    fig1.show(config=config)
    return


def add_plot(x, y, name, fig2):
    if name == 'g(x)':
        fig2.add_trace(
            go.Scatter(
                x=x,
                y=y,
                name=name,
                line=dict(width=4)))
        fig2['data'][0]['showlegend']=True
        fig2['data'][0]['name']=name
        fig2.update_layout(
            width=half_fig_width,
            height=full_fig_height)
    else:
        fig2.add_trace(
            go.Scatter(
                x=x,
                y=y,
                name=name,
                line=dict(width=3)))
    fig2.update_layout(
        legend=dict(
                x=1,
                y=1,
                font_size=26),
        xaxis={'visible': True, 'showticklabels': True},
        yaxis={'visible': True, 'showticklabels': True}
    )
    return

def plot_fig3():
    # Define GP models and fit prior on plots
    x = np.linspace(-5, 5, 100)
    g = np.sin(2 * x) + np.cos(1 * x)
    y = g + np.random.normal(size=len(x), scale=0.5)
    data = (x.reshape(-1, 1), y.reshape(-1, 1))
    kernel = gpflow.kernels.Matern32()

    color1 = plotly.colors.qualitative.Set2[0]
    color2 = plotly.colors.qualitative.Set2[1]
    color3 = plotly.colors.qualitative.Set2[2]

    # Set up plot and make plot
    fig3 = go.FigureWidget(
        plotly.tools.make_subplots(
            rows=1,
            cols=2,
            shared_yaxes=True,
            subplot_titles=("GP","SparseGP")))
    fig3.add_trace(
        go.Scatter(
            x=x, y=y,
            mode='markers',
            name='Observations',
            line=dict(color='black')),
        row=1, col=1
    );
    fig3.add_trace(
        go.Scatter(
            x=x, y=y,
            mode='markers',
            line=dict(color='black')),
        row=1, col=2,
    );
    fig3['data'][1]['showlegend'] = False

    # define gps
    gp = gpflow.models.GPR(data, kernel)
    gp.kernel.lengthscales.assign(10)
    opt_gp = gpflow.optimizers.Scipy()

    Z = np.linspace(-1.5, 1.5, 10).reshape(-1, 1)
    sgp = gpflow.models.SGPR(data, kernel, Z)
    sgp.kernel.lengthscales.assign(10)
    opt_sgp = gpflow.optimizers.Scipy()

    # make prior predictions
    msgp, vsgp = sgp.predict_y(x.reshape(-1, 1))
    musgp = msgp.numpy().ravel()
    stdsgp = vsgp.numpy().ravel() ** 0.5
    ysgpz = sgp.predict_y(sgp.inducing_variable.Z)[0].numpy().ravel()

    mgp, vgp = gp.predict_y(x.reshape(-1, 1))
    mugp = mgp.numpy().ravel()
    stdgp = vgp.numpy().ravel() ** 0.5


    # plot gp prior uncertainty
    fig3.add_trace(
        go.Scatter(
            x=x, y=mugp - stdgp,
            fill=None,
            line=dict(width=0)),
        row=1, col=1)
    fig3['data'][-1]['showlegend'] = False

    fig3.add_trace(
        go.Scatter(
            x=x,
            y=mugp + stdgp,
            fill='tonexty', # fill area between trace0 and trace1
            name='GP 95%',
            line_color=color1,
            line=dict(width=0)),
        row=1, col=1)

    # plot gp prior uncertainty
    fig3.add_trace(
        go.Scatter(
            x=x, y=musgp - stdsgp,
            fill=None,
            line=dict(width=0)),
        row=1, col=2)
    fig3['data'][-1]['showlegend'] = False
    fig3.add_trace(
        go.Scatter(
            x=x,
            y=musgp + stdsgp,
            fill='tonexty', # fill area between trace0 and trace1
            name='SparseGP 95%',
            line_color=color2,
            line=dict(width=0)),
        row=1, col=2)

    # Add mean lines
    fig3.add_trace(
        go.Scatter(
            x=x,
            y=mugp,
            line_color=color1),
        row=1, col=1)
    fig3['data'][-1]['showlegend'] = False

    fig3.add_trace(
        go.Scatter(
            x=x,
            y=musgp,
            line_color=color2),
        row=1, col=2)
    fig3['data'][-1]['showlegend'] = False

    fig3.add_trace(
        go.Scatter(
            x=Z.ravel(),
            y=ysgpz,
            name='Inducing points',
            marker_color=color3,
            mode="markers",
            marker_symbol='cross-thin',
            marker_line_color=color3,
            marker_line_width=3,
            marker_size=18,
        ),
        row=1, col=2)

    fig3.update_layout(
        width=full_fig_width,
        height=full_fig_height,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)');

    fig3.update_annotations(font_size=32)

    fig3.update_xaxes(range=[-5, 5])
    
    return fig3, gp, sgp, x, opt_gp, opt_sgp



def fit_fig3(fig3, gp, sgp, x, opt_gp, opt_sgp):
    for i in range(20):
        opt_logs = opt_gp.minimize(
            gp.training_loss, gp.trainable_variables, options=dict(maxiter=1)
        )
        opt_logs = opt_sgp.minimize(
            sgp.training_loss, sgp.trainable_variables, options=dict(maxiter=1)
        )
        
        msgp, vsgp = sgp.predict_y(x.reshape(-1, 1))
        musgp = msgp.numpy().ravel()
        stdsgp = vsgp.numpy().ravel() ** 0.5
        ysgpz = sgp.predict_y(sgp.inducing_variable.Z)[0].numpy().ravel()
        
        mgp, vgp = gp.predict_y(x.reshape(-1, 1))
        mugp = mgp.numpy().ravel()
        stdgp = vgp.numpy().ravel() ** 0.5
        
        fig3['data'][2]['y'] = mugp - stdgp
        fig3['data'][3]['y'] = mugp + stdgp
    
        fig3['data'][4]['y'] = musgp - stdsgp
        fig3['data'][5]['y'] = musgp + stdsgp
        
        fig3['data'][6]['y'] = mugp
        fig3['data'][7]['y'] = musgp
        
        fig3['data'][8]['x'] = sgp.inducing_variable.Z.numpy().ravel()
        fig3['data'][8]['y'] = ysgpz

    return

def plot_fig5():
    # Hide, run initially

    # Initialise figure
    subplots = plotly.tools.make_subplots(
        rows=2, cols=4,
        specs=[[{"rowspan":2, "colspan":2}, None, {}, {}],
               [None, None, {}, {}]],
        shared_yaxes=True,
        shared_xaxes = True,
        subplot_titles=(['Latent', 'CanESM5', 'CESM2-WACCM', 'KIOST-ESM', 'GISS-E2-1-G'])
        )
    fig5 = go.FigureWidget(subplots)
    fig5.update_annotations(font_size=26)

    # Colors
    color1 = plotly.colors.qualitative.Set2[0]
    color2 = plotly.colors.qualitative.Set2[1]
    color3 = plotly.colors.qualitative.Set2[2]
    color4 = plotly.colors.qualitative.Set2[3]
    color5 = plotly.colors.qualitative.Set2[4]

    # Initialise data for SHGP
    da = xr.open_dataarray('./../data/tas-ssp245.nc') - 273.15
    da = da[:, -365 * 3:]
    x_time = np.asarray([np.datetime64(t) for t in da.time.values])
    x = np.arange(x_time.size)
    x_cos = np.sin(2 * np.pi * da.time.dt.dayofyear / 365) # no leap calendars
    x_sin = np.cos(2 * np.pi * da.time.dt.dayofyear / 365)
    X = np.stack([x, x_cos, x_sin], axis=1)
    Y = da.values.T

    # Normalise data
    X_mean = np.mean(X, axis=0)
    X_std = np.std(X, axis=0)
    X_norm = (X - X_mean)/X_std
    Y_mean = np.mean(Y)
    Y_std = np.std(Y)
    Y_norm = (Y - Y_mean)/Y_std
    X_norm = X_norm.astype(np.float64)
    Y_norm = np.asarray(Y_norm, dtype=np.float64)

    # data
    data = (X_norm, Y_norm)

    # Initialise SHGP
    n_inducing = 50
    Z = np.linspace(np.min(X_norm, axis=0), np.max(X_norm, axis=0), n_inducing)
    kernel = gpflow.kernels.Matern32()

    shgp = SHGP(
        data,
        group_kernel=kernel,
        individual_kernel=kernel,
        inducing_points=Z)

    # Make predictions
    indi_preds = [shgp.predict_individual(X_norm, idx) for idx in range(len(da.realisation))]
    indi_preds = [(m.numpy().squeeze(), np.sqrt(v.numpy().squeeze())) for m,v in indi_preds]

    group_mean, group_var = shgp.predict_group(X_norm)
    group_mean = group_mean.numpy().squeeze()
    group_std = np.sqrt(group_var.numpy().squeeze())

    # Denormalise
    group_mean = (group_mean * Y_std + Y_mean)
    group_std = group_std * Y_std
    indi_preds = [(m * Y_std + Y_mean, v * Y_std) for m,v in indi_preds]

    # Plot data
    def plot_data(fig, y, row, col, color):
        fig.add_trace(
            go.Scatter(
                x=x_time, y=y,
                mode='markers',
                marker_size=2,
                line=dict(color=color)),
            row=row, col=col)
        return

    plot_data(fig5, Y[:, 0], 1, 3, 'black')
    plot_data(fig5, Y[:, 1], 1, 4, 'black')
    plot_data(fig5, Y[:, 2], 2, 3, 'black')
    plot_data(fig5, Y[:, 3], 2, 4, 'black')

    # Plot priors
    def plot_mean(fig, mean, color, row, col):
        fig.add_trace(
        go.Scatter(
                x=x_time,
                y=mean,
                line_color=color),
            row=row, col=col)
        return
    def plot_fill(fig, mean, std, color, row, col):
        fig.add_trace(
            go.Scatter(
                x=x_time,
                y=mean - 2 * std,
                fill=None,
                line=dict(width=0)),
            row=row, col=col)
        fig.add_trace(
            go.Scatter(
                x=x_time,
                y=mean + 2 * std,
                fill='tonexty', # fill area between trace0 and trace1
                line_color=color,
                line=dict(width=0)),
            row=row, col=col)
        return

    ## Latent
    plot_fill(fig5, group_mean, group_std, color1, 1, 1)
    plot_mean(fig5, group_mean, color1, 1, 1)

    ## CanESM5
    plot_fill(fig5, indi_preds[0][0], indi_preds[0][1], color2, 1, 3)
    plot_mean(fig5, indi_preds[0][0], color2, 1, 3)
    plot_mean(fig5, indi_preds[0][0], 'gray', 1, 1)


    ## WACCM
    plot_fill(fig5, indi_preds[1][0], indi_preds[1][1], color3, 1, 4)
    plot_mean(fig5, indi_preds[1][0], color3, 1, 4)
    plot_mean(fig5, indi_preds[1][0], 'gray', 1, 1)


    ## KIOST
    plot_fill(fig5, indi_preds[2][0], indi_preds[2][1], color4, 2, 3)
    plot_mean(fig5, indi_preds[2][0], color4, 2, 3)
    plot_mean(fig5, indi_preds[2][0], 'gray', 1, 1)


    ##GISS
    plot_fill(fig5, indi_preds[3][0], indi_preds[3][1], color5, 2, 4)
    plot_mean(fig5, indi_preds[3][0], color5, 2, 4)
    plot_mean(fig5, indi_preds[3][0], 'gray', 1, 1)


    # Extra plot config
    fig5.update_layout(
        yaxis_title='Surface temperature (°C)',
        width=full_fig_width * 1.15,
        height=full_fig_height,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(t=35),
        showlegend=False);

    fig5.update_xaxes(range=[x_time[int(-2.9 * 365)], x_time[int(-0.1 * 365)]]);
    fig5.update_yaxes(range=[-5, 30]);
    
    normalisation_vals = dict(
        X_mean=X_mean,
        X_std=X_std,
        X_norm=X_norm,
        Y_mean=Y_mean,
        Y_std=Y_std,
        Y_norm=Y_norm
    )
    
    return fig5, shgp, da, normalisation_vals

def fit_fig5(fig5, shgp, da, normalisation_vals, n_iters=10):
    params = dict()
    params['optim_nits'] = 1
    params['log_interval']= 200
    params['learning_rate'] = 0.2
    X_mean = normalisation_vals['X_mean']
    X_std = normalisation_vals['X_std']
    X_norm = normalisation_vals['X_norm']
    Y_mean = normalisation_vals['Y_mean']
    Y_std = normalisation_vals['Y_std']
    Y_norm = normalisation_vals['Y_norm']
    
    for i in trange(n_iters):
        shgp.fit(params, compile=False)

        # Make predictions
        indi_preds = [shgp.predict_individual(X_norm, idx) for idx in range(len(da.realisation))]
        indi_preds = [(m.numpy().squeeze(), np.sqrt(v.numpy().squeeze())) for m,v in indi_preds]

        group_mean, group_var = shgp.predict_group(X_norm)
        group_mean = group_mean.numpy().squeeze()
        group_std = np.sqrt(group_var.numpy().squeeze())

        # Denormalise
        group_mean = (group_mean * Y_std + Y_mean)
        group_std = group_std * Y_std
        indi_preds = [(m * Y_std + Y_mean, v * Y_std) for m,v in indi_preds]

        # Latent
        fig5['data'][4]['y'] = group_mean - group_std
        fig5['data'][5]['y'] = group_mean + group_std
        fig5['data'][6]['y'] = group_mean

        # Models 
        for i in range(len(indi_preds)):
            fig5['data'][7 + i * 4]['y'] = indi_preds[i][0] - indi_preds[i][1]
            fig5['data'][8 + i * 4]['y'] = indi_preds[i][0] + indi_preds[i][1]
            fig5['data'][9 + i * 4]['y'] = indi_preds[i][0]
            fig5['data'][10 + i * 4]['y'] = indi_preds[i][0]
            
    
    return