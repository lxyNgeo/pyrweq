# pyrweq 项目规范

Python 实现的修正风蚀方程（RWEQ）。用于科研数据处理与学术论文写作（导师：资源与环境专业，硕士研究生）。

## 项目结构约定

```
pyrweq/
├── src/pyrweq/
│   ├── core.py          # compute_rweq 主入口（因子并行、dask/numpy 后端分发）
│   ├── erosion.py       # calc_sl 公式
│   ├── io.py            # GeoTIFF 读写（read_raster / write_raster / load_inputs）
│   ├── classify.py      # 侵蚀强度分级（SL190-2007）
│   ├── sandfix.py       # 防风固沙量 G = SL_pot(C=1) - SL_actual
│   ├── stats.py         # 分区统计
│   ├── _types.py        # 类型别名（RasterInput / FactorArray / RasterioProfile）
│   └── factors/         # 各因子计算（纯函数，不读文件）
│       ├── weather.py   # WF
│       ├── erodibility.py  # EF
│       ├── crust.py     # SCF
│       ├── roughness.py # K'
│       ├── vegetation.py   # C
│       └── helpers.py   # 共用小函数（风速换算、空气密度、SW、SCF）
└── tests/               # unittest 风格（不用 pytest）
```

## 核心设计原则

1. **factors/ 是纯函数层**：只接受数组、返回数组，不读写文件、不抛业务日志以外的副作用。I/O 全在 io.py 和 core.py。
2. **数组类型透明**：因子函数签名用 `FactorArray`（numpy 或 dask 均可），内部运算必须同时兼容两者——禁止 `arr.shape = ...`（NumPy 2.5 弃用）、禁止原地修改、mask 赋值用 `np.where` 而不是 `result[mask] = ...`（后者在 dask 上不可用）。
3. **可选依赖零硬依赖**：dask 用 `type(arr).__module__.startswith("dask.array")` 检测 + try/except 导入，不 import 就崩。
4. **日志**：模块级 `logger = logging.getLogger(__name__)`，`pyrweq` 根 logger 带 NullHandler。警告用 `logger.warning`（异常阈值条件），常规进度用 `logger.info`。不要 print。

## 测试约定

- 测试用 **unittest** 风格（`import unittest` + `TestCase`），不用 pytest，因为当前环境没装 pytest。
- 运行：`cd /d/pyrweq && PYTHONPATH=src python -m unittest discover -s tests -v`
- 改动后必须跑全套测试，全部通过才算完成。
- dask 相关测试用 `@unittest.skipUnless(_HAS_DASK, ...)` 保护，未装 dask 时跳过而非失败。
- 新功能必须有对应测试（因子边界、集成、并行一致性、日志行为）。

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
