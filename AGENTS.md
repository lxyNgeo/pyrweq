# pyrweq 项目规范

Python 实现的修正风蚀方程（RWEQ）科学计算库，面向科研场景的栅格数据处理与论文结果产出。

## 项目结构约定

```
pyrweq/
├── src/pyrweq/
│   ├── core.py          # compute_rweq 主入口（因子并行、dask/numpy 后端分发）
│   ├── erosion.py       # calc_sl 公式（SL 单位 g/m）
│   ├── io.py            # GeoTIFF 读写（read_raster / write_raster / load_inputs / align_inputs）
│   ├── units.py         # 单位链转换（SL g/m ↔ 模数 t/(km²·a)，基于像元尺寸）
│   ├── classify.py      # 侵蚀强度分级（SL190-2007，阈值基于 t/(km²·a) 模数）
│   ├── sandfix.py       # 防风固沙量 G = SL_pot(C=1) - SL_actual（单位同 SL）
│   ├── validate.py      # 验证指标（r2/rmse/mae/bias/NSE）+ 站点采样
│   ├── sensitivity.py   # OAT 参数敏感性（弹性系数）
│   ├── stats.py         # 分区统计
│   ├── cli.py           # 命令行入口（compute/sandfix/classify/stats）
│   ├── _types.py        # 类型别名（RasterInput / FactorArray / RasterioProfile）
│   └── factors/         # 各因子计算（纯函数，不读文件）
│       ├── weather.py   # WF（支持 2D 单值 / 3D 观测序列 / 3D 频率分布风速）
│       ├── erodibility.py  # EF
│       ├── crust.py     # SCF
│       ├── roughness.py # K'
│       ├── vegetation.py   # C
│       └── helpers.py   # 共用小函数（风速换算、空气密度、SW、SCF）
├── examples/
│   ├── benchmark.py     # 性能基准（numpy/dask × seq/par）
│   └── rweq_demo.ipynb  # 全流程示例 notebook
└── tests/               # unittest 风格（不用 pytest）
```

## 核心设计原则

1. **factors/ 是纯函数层**：只接受数组、返回数组，不读写文件、不抛业务日志以外的副作用。I/O 全在 io.py 和 core.py。风速支持 2D 单值与 3D (k, rows, cols) 序列/频率分布；core.py 负责从 3D 输入提取空间 shape（[-2:]），其他因子始终 2D。
2. **数组类型透明**：因子函数签名用 `FactorArray`（numpy 或 dask 均可），内部运算必须同时兼容两者——禁止 `arr.shape = ...`（NumPy 2.5 弃用）、禁止原地修改、mask 赋值用 `np.where` 而不是 `result[mask] = ...`（后者在 dask 上不可用）。
3. **可选依赖零硬依赖**：dask 用 `type(arr).__module__.startswith("dask.array")` 检测 + try/except 导入，不 import 就崩。
4. **日志**：模块级 `logger = logging.getLogger(__name__)`，`pyrweq` 根 logger 带 NullHandler。警告用 `logger.warning`（异常阈值条件），常规进度用 `logger.info`。不要 print（CLI 除外）。
5. **nodata 语义**：读栅格默认 `masked=True`（nodata→NaN），NaN 穿过所有因子运算传播；写出时 `write_raster` 默认 NaN→nodata。统计必须用 nan* 函数。
6. **argparse help 字符串禁裸 `%`**（会被当格式占位符报错），用 "percent" 等措辞。

## 测试约定

- 测试用 **unittest** 风格（`import unittest` + `TestCase`），不用 pytest，因为当前环境没装 pytest。
- 运行：`cd /d/pyrweq && PYTHONPATH=src python -m unittest discover -s tests -v`
- 改动后必须跑全套测试（当前 120+），全部通过才算完成。提交前务必确认输出为 `OK`（勿只 grep Ran 行）。
- 单位链约定：calc_sl/compute_rweq 产出原生 g/m；分级/报告前用 units.g_per_m_to_t_per_km2 转 t/(km²·a)。新模块不得自带另一套单位语义。
- 周期约定：compute_rweq 默认 nd=15（RWEQ 半月）；compute_rweq_yearly 自动注入 nd=365.25/期数，除非显式覆盖。新增涉及周期的代码必须走同一路径。
- dask 相关测试用 `@unittest.skipUnless(_HAS_DASK, ...)` 保护，未装 dask 时跳过而非失败。
- 新功能必须有对应测试（因子边界、集成、并行一致性、日志行为、CLI、nodata）。
- notebook 改动后要验证可执行：提取 code cell 顺序执行（见会话记录）。

## 提交约定

- 提交信息用英文，动词开头：`feat:`, `fix:`, `refactor:`, `docs:`, `test:`。
- 改动前先 `git status` / `git diff` 自查；提交前跑一遍测试。
- 不提交：密钥、token、临时文件、大文件。
- git push / rebase / reset 等红线操作必须先问用户。

## 环境

- conda env: `geo-work`（Python 3.14 / numpy 2.5 / rasterio 1.5）。
- 包以 `PYTHONPATH=src` 方式使用，未 pip install -e 安装。
- 可选依赖：`dask`（并行后端）、`geopandas`（geo）、`matplotlib`（plot）。

## 已知问题

- rasterio 1.5 在 numpy 2.5 下 `src.read(1)` 触发 DeprecationWarning（rasterio 内部问题，非本项目代码）。
