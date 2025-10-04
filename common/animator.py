import matplotlib.pyplot as plt


class Animator:
    @staticmethod
    def enable_interactive():
        """开启matplotlib交互模式"""
        plt.ion()
    
    @staticmethod
    def disable_interactive():
        # 关闭matplotlib交互模式
        plt.ioff()
        # 阻塞, 等待用户关闭图表窗口
        plt.show()

    def set_axes(self, axes):
        # 初始化默认参数
        xlabel = getattr(axes, '_xlabel', None)
        ylabel = getattr(axes, '_ylabel', None)
        xlim = getattr(axes, '_xlim', None)
        ylim = getattr(axes, '_ylim', None)
        xscale = getattr(axes, '_xscale', 'linear')
        yscale = getattr(axes, '_yscale', 'linear')
        legend = axes.get_legend_handles_labels()[1] if axes.get_legend_handles_labels()[1] else None

        if xlabel is not None:
            axes.set_xlabel(xlabel)  # 设置x轴标签
        if ylabel is not None:
            axes.set_ylabel(ylabel)  # 设置y轴标签
        axes.set_xscale(xscale)  # 设置x轴刻度类型
        axes.set_yscale(yscale)  # 设置y轴刻度类型
        if xlim is not None:
            axes.set_xlim(xlim)      # 设置x轴取值范围
        if ylim is not None:
            axes.set_ylim(ylim)      # 设置y轴取值范围
        if legend:
            axes.legend(legend)  # 设置图例
        axes.grid()              # 显示网格线


    
    def __init__(self, xlabel=None, ylabel=None, legend=None, xlim=None,
                 ylim=None, xscale='linear', yscale='linear',
                 fmts=('-', 'm--', 'g-.', 'r:'), nrows=1, ncols=1,
                 figsize=(3.5, 2.5)):
        # 增量地绘制多条线
        if legend is None:
             # 如果没有提供图例，默认使用空列表
            legend = []
         # fig是 matplotlib.pyplot.figure 类的实例, 表示整个图表窗口
         # axes是 matplotlib.axes._subplots.AxesSubplot 类的实例, 表示带坐标的子图，二维 NumPy 数组
        self.fig, self.axes = plt.subplots(nrows, ncols, figsize=figsize)
        if nrows * ncols == 1:
             # 如果只有一个子图，将其转换为列表形式，按行优先顺序排列
            self.axes = [self.axes, ]

        # 存储绘制的数据线的坐标点和格式
        # X(N,G)
        # Y(N,G)
        # fmts(G)
        self.X, self.Y, self.fmts = None, None, fmts
    

    # x(N,G)
    # y(N,G)
    def add(self, x, y):
        # 1.准备数据
        if not hasattr(y, "__len__"):
            # 如果y不是一个序列, 则将其转换为包含一个元素的列表
            # y(1) -> y(N)
            y = [y]
        n = len(y) # 获取y的长度, 即有多少条线需要绘制
        if not hasattr(x, "__len__"):
            # 如果x不是一个序列, 则将其转换为包含n个元素的列表, 每个元素都等于x
            # 复制n次, 得到一个包含n个元素的列表, 每个元素都等于x
            # x(1) -> x(N)
            x = [x] * n

        if not self.X:
            # 如果X为空, 则初始化一个包含n个空列表的列表, 每个空列表对应一条线
            # X(N,G)
            self.X = [[] for _ in range(n)]
        if not self.Y:
            # 如果Y为空, 则初始化一个包含n个空列表的列表, 每个空列表对应一条线
            # Y(N,G)
            self.Y = [[] for _ in range(n)]

        # x(N)
        # y(N)
        # zip打包优先规则：先打包x的第一个元素和y的第一个元素, 然后打包x的第二个元素和y的第二个元素, 以此类推
        # a(1)
        # b(1)
        for i, (a, b) in enumerate(zip(x, y)): # 循环N次
            if a is not None and b is not None:
                # X(N,G)
                # Y(N,G)
                self.X[i].append(a) # 追加x坐标到第i条线的列表中
                self.Y[i].append(b) # 追加y坐标到第i条线的列表中


        # 2.清空当前子图（ self.axes[0].cla() ）
        # 3.重新绘制所有数据线
        # 4.重新配置坐标轴
        # 5.显示更新后的图表
        self.axes[0].cla()
        # zip行打包
        # self.X(N,G)
        # self.Y(N,G)
        # self.fmts(N)
        # x(G)
        # y(G)
        # fmt(1)
        for x, y, fmt in zip(self.X, self.Y, self.fmts):
            # 在同一个子图中绘制多条线
            lines= self.axes[0].plot(x, y, fmt)

        # 更新子图的坐标轴配置
        self.set_axes(self.axes[0])
        
        # 刷新图表, 并暂停0.01秒, 以便用户可以看到更新后的图表
        plt.draw()
        plt.pause(0.01)


        # # 刷新子图, 并暂停0.01秒, 以便用户可以看到更新后的图表
        # self.axes[0].figure.canvas.draw()
        # plt.pause(0.01)