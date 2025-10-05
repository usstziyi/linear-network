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

    def __init__(self, xlabel=None, ylabel=None, legend=None, xlim=None,
                 ylim=None, xscale='linear', yscale='linear',
                 fmts=('-', 'm--', 'g-.', 'r:'), nrows=1, ncols=1,
                 figsize=(3.5, 2.5)):
        if legend is None:
            legend = []

        self.fig, self.axes = plt.subplots(nrows, ncols, figsize=figsize)
        if nrows * ncols == 1:
            self.axes = [self.axes]

        # 初始化数据容器
        self.X = [[] for _ in range(len(fmts))]
        self.Y = [[] for _ in range(len(fmts))]

        # 创建空的线条对象并保存
        self.lines = []
        for fmt in fmts:
            # 创建空的线条对象并添加到列表中
            # 注意: plot返回的是一个元组, 我们只需要第一个元素
            line, = self.axes[0].plot([], [], fmt)
            self.lines.append(line)

        # 设置坐标轴属性
        ax = self.axes[0]
        if xlabel:
            ax.set_xlabel(xlabel)
        if ylabel:
            ax.set_ylabel(ylabel)
        ax.set_xscale(xscale)
        ax.set_yscale(yscale)
        if xlim is not None:
            ax.set_xlim(xlim)
        if ylim is not None:
            ax.set_ylim(ylim)
        if legend:
            ax.legend(legend)
        ax.grid()

    def add(self, x, y):
        # 标准化输入
        if not hasattr(y, "__len__"):
            y = [y]
        n = len(y)
        if not hasattr(x, "__len__"):
            x = [x] * n

        # 扩展 X/Y 列表（如果需要）
        while len(self.X) < n:
            self.X.append([])
            self.Y.append([])
            # 如果 fmts 不够，复用最后一个格式
            fmt = self.fmts[min(len(self.lines), len(self.fmts) - 1)]
            line, = self.axes[0].plot([], [], fmt)
            self.lines.append(line)

        # 追加新数据点
        for i in range(n):
            if x[i] is not None and y[i] is not None:
                self.X[i].append(x[i])
                self.Y[i].append(y[i])
                # 更新对应线条的数据
                self.lines[i].set_data(self.X[i], self.Y[i])


        # 后续动态扩展（不依赖 autoscale）
        current_max_x = max(x for row in self.X for x in row)
        current_max_y = max(y for row in self.Y for y in row)
        self.axes[0].set_xlim(right=max(self.axes[0].get_xlim()[1], current_max_x))
        self.axes[0].set_ylim(top=max(self.axes[0].get_ylim()[1], current_max_y))

        # 刷新
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        plt.pause(0.01)