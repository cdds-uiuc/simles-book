#!/usr/bin/env python
# coding: utf-8

# In[2]:


from xmovie import Movie
import xarray as xr

# load test dataset
ds = xr.tutorial.open_dataset('air_temperature').isel(time=slice(0,150))
# create movie object
mov = Movie(ds.air)

def custom_plotfunc(ds, fig, tt):
    # Define station location for timeseries
    station = dict(x=100, y=150)
    ds_station = ds.sel(**station)

    (ax1, ax2) = fig.subplots(ncols=2)
    
    # Map axis
    # Colorlimits need to be fixed or your video is going to cause seizures.
    # This is the only modification from the code above!
    ds.isel(time=tt).plot(ax=ax1, vmin=ds.min(), vmax=ds.max(), cmap='RdBu_r')
    ax1.plot(station['x'], station['y'], marker='*', color='k' ,markersize=15)
    ax1.text(station['x']+4, station['y']+4, 'Station', color='k' )
    ax1.set_aspect(1)
    ax1.set_facecolor('0.5')
    ax1.set_title('');

    # Time series
    ds_station.isel(time=slice(0,tt+1)).plot.line(ax=ax2, x='time')
    ax2.set_xlim(ds.time.min().data, ds.time.max().data)
    ax2.set_ylim(ds_station.min(), ds_station.max())
    ax2.set_title('Data at station');

    fig.subplots_adjust(wspace=0.6)
    return None, None #This is not strictly necessary, but otherwise a warning will be raised.  


mov_custom = Movie(ds.air, custom_plotfunc)
mov_custom.preview(30)


# In[11]:


# importing matplot lib
import matplotlib.pyplot as plt
import numpy as np
 
# importing movie py libraries
from moviepy.editor import VideoClip
from moviepy.video.io.bindings import mplfig_to_npimage
 
# numpy array
x = np.linspace(-2, 2, 200)
 
# duration of the video
duration = 2
 
# matplot subplot
fig, ax = plt.subplots()
 
# method to get frames
def make_frame(t):
     
    # clear
    ax.clear()
     
    # plotting line
    ax.plot(x, np.sinc(x**2) + np.sin(x + 2 * np.pi / duration * t), lw = 3)
    ax.set_ylim(-1.5, 2.5)
     
    # returning mumpy image
    return mplfig_to_npimage(fig)
 
# creating animation
animation = VideoClip(make_frame, duration = duration)
 
# displaying animation with auto play and looping
fps=30
fps
animation.ipython_display(fps = np.real(30), loop = True, autoplay = True)


# In[14]:


animation.


# In[10]:


fps=


# In[ ]:




