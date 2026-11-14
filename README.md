# 大模型实验作业八：自制语言模型知识蒸馏（2024312149 陈道延）

- 学号：2024312149　姓名：陈道延　班级：自动化四班
- 硬件：Intel CPU（`torch.set_num_threads(min(12, 核数))`），**纯 CPU，无 GPU**
- 报告：`2024312149_陈道延_实验作业八.docx`（6 页 / 12 表 / 1 图）
- 仓库：https://github.com/hitcdy/lab08-lm-distill

做法：语料完全自造（模板化古诗文，3000 行训练 + 400 行留出，零外部数据），
先用一个 0.737 M 参数的小 GPT 训成教师，再把教师的知识蒸馏到一个 0.147 M 的学生
（`loss = (1-α)·CE + α·T²·KL(student/T ‖ teacher/T)`），与同结构、同步数、同种子的
"从零训练"学生做公平对照，最后做 4×4 的 T×α 网格消融。

## 一、实测结果（CPU，350 iter，seed 7，T=4，α=0.5）

| 模型 | 参数量 | 压缩比 | 留出集 loss | PPL | 与教师一致率 | 训练耗时 |
| --- | --- | --- | --- | --- | --- | --- |
| 教师（teacher） | 0.737 M | — | **0.8614** | 2.37 | — | 240 iter |
| 从零学生（scratch） | 0.147 M | 5.0× | **0.8984** | 2.46 | — | 9.8 s |
| 蒸馏学生（KD） | 0.147 M | 5.0× | 1.0051 | 2.73 | 0.873 | 16.9 s |

- 三组同数据、同 block=96 / batch=24 / lr=3e-3 / seed=7，只差"是否蒸馏"。
- **本次蒸馏没有跑赢从零训练**（相对变化 −11.9%），原因见第七节，不粉饰。
- T×α 消融（16 组，每组 70 iter）：最优 **T=1、α=0.7 → loss 1.4843**；最差 **T=8、α=0.9 → 2.149**。
  完整网格见 `runs_distill/results_ablate.json`（表 7），规律：T=1 近似 one-hot 等于没蒸馏；
  T∈[2,4]、α∈[0.5,0.7] 才带上结构信息；α→0.9 时硬标签约束丢失，loss 回升。

结果图：`runs_distill/curves.png`（教师 / 从零 / 蒸馏三条训练与验证损失曲线）。

## 二、环境与依赖

| 项 | 值 |
| --- | --- |
| OS / Python | Windows 11 / Python 3.14（CPU 版 torch，`--index-url https://download.pytorch.org/whl/cpu`） |
| 依赖 | `torch`（≥2.x）、标准库 `argparse/json/math/os/random/time`；无第三方数据与预训练权重 |
| 线程 | `OMP_NUM_THREADS=12`（脚本内 `torch.set_num_threads` 已限 12） |
| 数据 | 脚本内 `build_corpus()` 现场生成，`vocab=322`，`train_chars=52456`，`val_chars=7387` |
| 模型 | 教师 2 层 / 4 头 / d=160；学生 2 层 / 4 头 / d=64；block=96 |

## 三、目录结构

```
lab08_lm_distill/
├── distill_lm.py                              # 全部实验：数据/模型/教师/蒸馏/消融，单文件
├── fill_report_hw8.py                         # 把实测数据、曲线、结论填进课程模板 → 报告 docx
├── runs_distill/
│   ├── teacher.pt                             # 教师权重（2.8 MB）
│   ├── results_teacher.json                   # 教师：0.737 M / val_loss 0.8614 / ppl 2.37
│   ├── results_all.json                       # 从零 vs 蒸馏（iters=350）
│   ├── results_ablate.json                    # 16 组 T×α 网格 + best
│   └── curves.png                             # 三条损失曲线
├── run_all.log / run_ablate.log               # 两种模式的控制台输出存档
├── 2024312149_陈道延_实验作业八.docx           # 实验报告（成品）
├── 作业八_自制语言模型知识蒸馏_实验指南.pdf      # 课程下发的实验指南
└── 作业八_自制语言模型知识蒸馏_实验报告（空白模板）.docx
```

## 四、一键复现

```bash
# 0) 装 CPU 版 torch（务必带 --index-url，否则会拉 2 GB+ 的 CUDA 版）
pip install torch --index-url https://download.pytorch.org/whl/cpu

# 1) 训练教师（240 iter，约 1~2 min，产出 runs_distill/teacher.pt + results_teacher.json）
python distill_lm.py --mode teacher

# 2) 从零基线 vs 蒸馏（350 iter；产出 results_all.json）
python distill_lm.py --mode all --iters 350

# 3) T×α 消融（4×4=16 组，每组 70 iter；产出 results_ablate.json）
python distill_lm.py --mode ablate --abl-iters 70

# 4) 自检（30 s，只验证前向/反向能跑通，不写正式结果）
python distill_lm.py --mode quick
```

说明：`teacher.pt` 已在仓库内，重跑第 2、3 步会直接复用；若要重训教师请删掉 `runs_distill/teacher.pt`
或加 `--retrain`（`--t-iters` 可改教师步数）。第 2、3 步会覆盖同名 JSON，重跑前建议先备份 `runs_distill/`。
报告由 `python fill_report_hw8.py` 依据上述 JSON + `curves.png` 重新生成。

## 五、KD 低于从零的 11.9%：原因分析

1. **本机迭代太快，学生早已不欠拟合**：350 步只要 9.8 s（指南同设置约 39 s，快约一个数量级），
   而蒸馏的收益主要出现在"学生容量/步数不足、还在欠拟合"的区间；一旦学生自己就能把训练集吃透，
   软标签的边际收益就消失，KL 项反成了额外约束。
2. **教师太"尖"，可教的暗知识有限**：教师只训 240 步就把留出集压到 0.8614（指南 0.9986），
   T=4 时软标签已接近 one-hot，分布里的类间相似性信息很少；此时 KL 更多是在让学生模仿教师的
   个别错误（教师自身 loss 0.8614 并非 0），于是学生反而被拉低。
3. 结论与改进方向：在 CPU 这种"步数充足"的条件下，应改用**更小的学生 / 更少的步数 / 更高温度
   （T=4~8）+ 中等权重（α=0.5~0.7）**，或先弱化教师（少训几十步）再蒸馏——消融里 T=1、α=0.7
   最优也侧面说明"硬标签仍占主导时蒸馏才稳"。

## 六、结论

- 蒸馏链路（教师→软标签→T² 加权 KL + 硬标签 CE→学生）在纯 CPU 上可完整复现，单次实验分钟级。
- 小模型 + 充足步数时，KD 不一定优于从零训练；**蒸馏的收益依赖"学生欠拟合"这一前提**。
- T×α 存在明显最优区（本机 T=1、α=0.7 最低），温度过高、权重过大都会让硬标签失效而变差。
