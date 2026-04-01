#!/Library/Frameworks/Python.framework/Versions/3.7/bin/python3
# -*- coding:utf-8 -*-

'''
Copyright (C) 2018  CCLI GROUP, PKU
Copyright (C) 2018  Tan Wangshu
NAME
        PlotStyle.py
PURPOSE

PROGRAMMER(S)
        Wangshu TAN
REVISION HISTORY

REFERENCES

----------------------------------------------------------
This script was created at 2018-12-05 13:46
If you have any question, please email: tanws@pku.edu.cn
----------------------------------------------------------
'''

import numpy as np
from matplotlib.ticker import MultipleLocator, AutoMinorLocator

__all__ = ['set_axis']

def set_axis(ax,xminor=False,yminor=False,**kwargs):
    '''Use this function to set major or minor ticks and linewidth of axis

    ---INPUT----
        ---args----
        ax                  : axes of which you want to adjust
        ---kwargs----
        xminor              : minor space of xaxis. can either be float or bool. default: False
        yminor              : minor space of yaxis. can either be float or bool. default: False
        xmajor              : major space of xaxis
        ymajor              : major space of yaxis
        tick_length         : length of ticks
        tick_width          : width of ticks
        xtick_length        : length of ticks of xaxis
        xtick_width         : width of ticks of xaxis
        ytick_length        : length of ticks of yaxis
        ytick_width         : width of ticks of yaxis
        major_length        : length of major ticks
        major_width         : width of major ticks
        minor_length        : length of minor ticks
        minor_width         : width of minor ticks
        xmajor_length       : length of major ticks of xaxis
        xmajor_width        : width of major ticks of xaxis
        xminor_length       : length of minor ticks of xaxis
        xminor_width        : width of minor ticks of xaxis
        ymajor_length       : length of major ticks of yaxis
        ymajor_width        : width of major ticks of yaxis
        yminor_length       : length of minor ticks of yaxis
        yminor_width        : width of minor ticks of yaxis
        ticklabel_fontsize  : fontsize of ticklabel
        ticklabel_weight    : weight of ticklabel
        xticklabel_fontsize : fontsize of xticklabel
        xticklabel_weight   : weight of xticklabel
        yticklabel_fontsize : fontsize of yticklabel
        yticklabel_weight   : weight of yticklabel
        axis_lw             : linewidth of axis.
        font_family         : font family name (e.g., 'serif', 'sans-serif')
        font_name           : specific font name (e.g., 'Times New Roman')
        label_fontsize      : fontsize of axis labels
        title_fontsize      : fontsize of axis title
    '''
    # 设置字体名称
    if 'font_name' in kwargs:
        font_name = kwargs['font_name']

        # 设置刻度标签字体
        for label in ax.get_xticklabels():
            label.set_fontname(font_name)
        for label in ax.get_yticklabels():
            label.set_fontname(font_name)

        # 设置轴标签字体
        if ax.xaxis.label:
            ax.xaxis.label.set_fontname(font_name)
        if ax.yaxis.label:
            ax.yaxis.label.set_fontname(font_name)

        # 设置标题字体
        if ax.title:
            ax.title.set_fontname(font_name)

    if xminor:
        try:
            if str(xminor)!='True':
                ax.xaxis.set_minor_locator(MultipleLocator(float(xminor)))
            else:
                ax.xaxis.set_minor_locator(AutoMinorLocator())
        except:
            ax.xaxis.set_minor_locator(AutoMinorLocator())
    
    if yminor:
        try:
            if str(yminor)!='True':
                ax.yaxis.set_minor_locator(MultipleLocator(float(yminor)))
            else:
                ax.yaxis.set_minor_locator(AutoMinorLocator())
        except:
            ax.yaxis.set_minor_locator(AutoMinorLocator())

    if 'xmajor' in kwargs:
        xmajor = kwargs['xmajor']
        ax.xaxis.set_major_locator(MultipleLocator(float(xmajor)))

    if 'ymajor' in kwargs:
        ymajor = kwargs['ymajor']
        ax.yaxis.set_major_locator(MultipleLocator(float(ymajor)))

    if 'tick_length' in kwargs:
        tick_length = kwargs['tick_length']
        ax.tick_params(which='both',length=tick_length)

    if 'tick_width' in kwargs:
        tick_width = kwargs['tick_width']
        ax.tick_params(which='both',width=tick_width)

    if 'xtick_length' in kwargs:
        xtick_length = kwargs['xtick_length']
        ax.tick_params(axis='x',which='both',length=xtick_length)

    if 'xtick_width' in kwargs:
        xtick_width = kwargs['xtick_width']
        ax.tick_params(axis='x',which='both',width=xtick_width)

    if 'ytick_length' in kwargs:
        ytick_length = kwargs['ytick_length']
        ax.tick_params(axis='y',which='both',length=ytick_length)

    if 'ytick_width' in kwargs:
        ytick_width = kwargs['ytick_width']
        ax.tick_params(axis='y',which='both',width=ytick_width)

    if 'major_length' in kwargs:
        major_length = kwargs['major_length']
        ax.tick_params(which='major',length=major_length)

    if 'major_width' in kwargs:
        major_width = kwargs['major_width']
        ax.tick_params(which='major',width=major_width)

    if 'minor_length' in kwargs:
        minor_length = kwargs['minor_length']
        ax.tick_params(which='minor',length=minor_length)

    if 'minor_width' in kwargs:
        minor_width = kwargs['minor_width']
        ax.tick_params(which='minor',width=minor_width)

    if 'xmajor_length' in kwargs:
        xmajor_length = kwargs['xmajor_length']
        ax.tick_params(axis='x',which='major',length=xmajor_length)

    if 'xmajor_width' in kwargs:
        xmajor_width = kwargs['xmajor_width']
        ax.tick_params(axis='x',which='major',width=xmajor_width)

    if 'xminor_length' in kwargs:
        xminor_length = kwargs['xminor_length']
        ax.tick_params(axis='x',which='minor',length=xminor_length)

    if 'xminor_width' in kwargs:
        xminor_width = kwargs['xminor_width']
        ax.tick_params(axis='x',which='minor',width=xminor_width)

    if 'ymajor_length' in kwargs:
        ymajor_length = kwargs['ymajor_length']
        ax.tick_params(axis='y',which='major',length=ymajor_length)

    if 'ymajor_width' in kwargs:
        ymajor_width = kwargs['ymajor_width']
        ax.tick_params(axis='y',which='major',width=ymajor_width)

    if 'yminor_length' in kwargs:
        yminor_length = kwargs['yminor_length']
        ax.tick_params(axis='y',which='minor',length=yminor_length)

    if 'yminor_width' in kwargs:
        yminor_width = kwargs['yminor_width']
        ax.tick_params(axis='y',which='minor',width=yminor_width)
    
    if 'ticklabel_fontsize' in kwargs:
        ticklabel_fontsize = kwargs['ticklabel_fontsize']
        for label in ax.get_xticklabels():
            label.set_fontsize(ticklabel_fontsize)
        for label in ax.get_yticklabels():
            label.set_fontsize(ticklabel_fontsize)
    
    if 'ticklabel_weight' in kwargs:
        ticklabel_weight = kwargs['ticklabel_weight']
        for label in ax.get_xticklabels():
            label.set_weight(ticklabel_weight)
        for label in ax.get_yticklabels():
            label.set_weight(ticklabel_weight)
    
    if 'xticklabel_fontsize' in kwargs:
        xticklabel_fontsize = kwargs['xticklabel_fontsize']
        for label in ax.get_xticklabels():
            label.set_fontsize(xticklabel_fontsize)

    if 'xticklabel_weight' in kwargs:
        xticklabel_weight = kwargs['xticklabel_weight']
        for label in ax.get_xticklabels():
            label.set_weight(xticklabel_weight)

    if 'yticklabel_fontsize' in kwargs:
        yticklabel_fontsize = kwargs['yticklabel_fontsize']
        for label in ax.get_yticklabels():
            label.set_fontsize(yticklabel_fontsize)

    if 'yticklabel_weight' in kwargs:
        yticklabel_weight = kwargs['yticklabel_weight']
        for label in ax.get_yticklabels():
            label.set_weight(yticklabel_weight)

    if 'label_fontsize' in kwargs:
        label_fontsize = kwargs['label_fontsize']
        if ax.xaxis.label:
            ax.xaxis.label.set_fontsize(label_fontsize)
        if ax.yaxis.label:
            ax.yaxis.label.set_fontsize(label_fontsize)

    if 'title_fontsize' in kwargs:
        title_fontsize = kwargs['title_fontsize']
        if ax.title:
            ax.title.set_fontsize(title_fontsize)

    if 'axis_lw' in kwargs:
        axis_lw = kwargs['axis_lw']
        for axis in ['top','bottom','left','right']:
              ax.spines[axis].set_linewidth(axis_lw)

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(16,9))
    ax = plt.gca()
    # ax.plot([1,2,3,4,5],[1,4,9,16,25])
    ax.set_xlabel('X Axis Label')
    ax.set_ylabel('Y Axis Label')
    ax.set_title('Title with Times New Roman Font')
    set_axis(ax,
             xminor=0.15,
             yminor=0.05,
             axis_lw=1.5,
             minor_width=1.5,
             minor_length=5,
             major_length=8,
             major_width=1.5,
             font_name='Times New Roman',
             font_family='serif',
             ticklabel_fontsize=12,
             label_fontsize=14,
             title_fontsize=16)
    plt.show()
