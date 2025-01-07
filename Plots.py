import seaborn as sns
import matplotlib.pyplot as plt


sns.set_style('darkgrid')
sns.color_palette(palette='RdYlGn')


# Function for plotting installed capacity per technology
def cap_plot(data):
    sns.barplot(x='Technology', y='Capacity [kW]', data=data, errorbar=None, palette='hot',
                edgecolor='black').set(title='Installed Capacities per technology')
    plt.show()


# Function for plotting plant dispatch over time
def plot_operation_vs_time(df, time_start=0, time_stop = 8759):

        color_h2 = 'forestgreen'
        color_x_el_sold = 'orange'
        color_market_price = 'black'
        color_x_el_available = 'c'
        ax1 = sns.scatterplot(x=df.time[time_start:time_stop], y=df.H2_prod[time_start:time_stop], color=color_h2)
        ax2 = ax1.twinx()
        sns.scatterplot(x=df.time[time_start:time_stop], y=df.El_sold[time_start:time_stop], ax=ax2, color=color_x_el_sold)
        #sns.scatterplot(x=df.time[time_start:time_stop], y=df.Electrcitiy_available[time_start:time_stop], ax=ax2,color=color_x_el_available)
        ax1.set_ylabel('H2 produced [kg/h]', color=color_h2)
        ax1.set_xlabel('Time [h]')
        ax1.set_title('Operation of the plant', fontdict = { 'fontsize': 14, 'verticalalignment': 'center', 'horizontalalignment': 'center'})
        ax2.set_ylabel('$Electricity sold [MWh]$', color=color_x_el_sold)
        ax3 = ax1.twinx()
        ax3.set_ylabel("Market price [€/MWh]", color = color_market_price)
        sns.scatterplot(x=df.time[time_start:time_stop], y = df.Market_price[time_start:time_stop], ax=ax3, color=color_market_price)

        ax3.spines['right'].set_position(('outward', 50))
        #ax3.spines['right'].set_position(('axes', 1.15))

        ax1.tick_params(axis='y', colors=color_h2)
        ax2.tick_params(axis='y', colors=color_x_el_sold)
        ax3.tick_params(axis='y', colors=color_market_price)

        ax2.spines['right'].set_color(color_x_el_sold)
        ax3.spines['right'].set_color(color_market_price)
        ax3.spines['left'].set_color(color_market_price)

        plt.show()


