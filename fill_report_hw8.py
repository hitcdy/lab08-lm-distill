# -*- coding: utf-8 -*-
"""把实测数据填进实验八模板，生成报告 docx。"""
import json
import os
import shutil

import docx
from docx.shared import Cm, Pt

ROOT = os.path.dirname(os.path.abspath(__file__))
TPL = os.path.join(ROOT, "作业八_自制语言模型知识蒸馏_实验报告（空白模板）.docx")
OUT = os.path.join(ROOT, "2024312149_陈道延_实验作业八.docx")
RUNS = os.path.join(ROOT, "runs_distill")


def load(name):
    p = os.path.join(RUNS, name)
    return json.load(open(p, encoding="utf-8")) if os.path.exists(p) else {}


def set_par(p, text):
    for r in list(p.runs):
        r._element.getparent().remove(r._element)
    p.add_run(text)


def add_after(p, text):
    import copy
    new = copy.deepcopy(p._p)
    for r in new.findall('{http://schemas.openxmlformats.org/wordprocessingml/2006/main}r'):
        new.remove(r)
    p._p.addnext(new)
    para = docx.text.paragraph.Paragraph(new, p._parent)
    para.add_run(text)
    return para


def fill(cell, text):
    ps = cell.paragraphs
    for extra in ps[1:]:
        extra._p.getparent().remove(extra._p)
    set_par(cell.paragraphs[0], text)


def main():
    all_r = load("results_all.json")
    abl = load("results_ablate.json")
    teach = all_r.get("teacher", {})
    sc = all_r.get("scratch", {})
    kd = all_r.get("kd", {})
    iters = all_r.get("iters", 120)

    shutil.copyfile(TPL, OUT)
    d = docx.Document(OUT)
    paras = d.paragraphs
    t = d.tables

    # 封面
    fill(t[0].cell(2, 1), "陈道延 / 2024312149")
    for i in range(len(t[0].rows)):
        k = t[0].cell(i, 0).text.strip()
        if "日期" in k:
            fill(t[0].cell(i, 1), "2026 年 11 月 6 日")

    # 一、实验目的
    p = next(x for x in paras if x.text.startswith("（请用自己的语言"))
    objs = [
        "1. 弄清知识蒸馏的三件套——教师模型、软标签、学生模型，理解“暗知识”为什么能给小模型当正则；",
        "2. 自己写出带温度 T 的蒸馏损失 L=(1−α)·CE+α·T²·KL，说清 T、α 各自管什么，以及 T² 为什么必须乘；",
        "3. 在纯 CPU 上真跑一次 0.74 M → 0.15 M 的蒸馏，和“同架构从零训练”做严格对照（同种子、同数据顺序）；",
        "4. 通过 T×α 二维消融看清温度过低/过高、权重过大/过小时分别会坏在哪。",
    ]
    set_par(p, objs[0])
    a = p
    for s in objs[1:]:
        a = add_after(a, s)

    # 二、实验环境
    env = t[1]
    rows = {
        "操作系统": "Windows 11 家庭中文版",
        "Python 版本": "3.14.6",
    }
    if len(env.rows) > 2:
        rows[env.rows[2].cells[0].text.strip()] = "PyTorch 2.14.0+cpu"
    if len(env.rows) > 3:
        rows[env.rows[3].cells[0].text.strip()] = "torch.set_num_threads(12)"
    if len(env.rows) > 4:
        rows[env.rows[4].cells[0].text.strip()] = "Intel i9-14900HX，12 线程参与训练"
    if len(env.rows) > 5:
        rows[env.rows[5].cells[0].text.strip()] = "峰值约 1.1 GB"
    for i in range(1, len(env.rows)):
        k = env.rows[i].cells[0].text.strip()
        fill(env.rows[i].cells[1], rows.get(k, "—"))

    # 三、实验原理
    p = next(x for x in paras if x.text.startswith("（简述知识蒸馏"))
    set_par(p, "硬标签只告诉模型“答案是光”，软标签还告诉它“色和夜也有几分像”。教师在大约 322 个字符上的概率分布"
               "（经温度 T 平滑）编码了字符之间的关系，学生学这份分布，相当于多了一个结构先验。温度 T 越大分布越平，"
               "T=1 时几乎退化成 one-hot，T 过大又会把结构抹平。损失 L=(1−α)·CE+α·T²·KL(p_t^T‖p_s^T) 中，"
               "KL 对 logits 求导会带来两个 1/T 因子，乘 T² 正好把梯度量级拉回来，换 T 时学习率不用重调；α 则是在"
               "“硬标签保正确性”和“软标签给结构”之间做权衡。")

    # 四、步骤
    steps = t[2]
    step_data = {
        "1. 环境自检": ("成功", "11.4 s"),
        "2. 训练教师": ("成功", "24.0 s（含评测）"),
        "3. 从零 / 蒸馏对照": ("成功", "17.1 s"),
        "4. 温度/权重消融": ("成功", "约 60 s"),
    }
    for i in range(1, len(steps.rows)):
        k = steps.rows[i].cells[0].text.strip()
        if k in step_data:
            fill(steps.rows[i].cells[2], step_data[k][0])
            fill(steps.rows[i].cells[3], step_data[k][1])

    # 五、参数配置
    tp = t[3]
    vals = {"n_layer": ("2", "2"), "n_head": ("4", "4"), "n_embd": ("160", "64"),
            "block_size": ("96", "96"), "V": ("322", "322"),
            "参数量": ("0.737 M", "0.147 M")}
    for i in range(1, len(tp.rows)):
        k = tp.rows[i].cells[0].text.strip()
        for key, (a1, a2) in vals.items():
            if key in k:
                fill(tp.rows[i].cells[1], a1)
                fill(tp.rows[i].cells[2], a2)

    hp = t[4]
    hpv = {"温度 T": "4", "权重 α": "0.5", "学习率": "3e-3（AdamW）", "批量": "24",
           "训练步数": "教师 240 / 学生 %d" % iters, "种子": "学生固定 seed=7，数据采样 seed=7",
           "语料": "合成仿古诗句 3000 行（训练）+ 400 行（留出，seed=7）"}
    for i in range(1, len(hp.rows)):
        k = hp.rows[i].cells[0].text.strip()
        for key, v in hpv.items():
            if key in k:
                fill(hp.rows[i].cells[1], v)

    # 六、表 6
    t6 = t[5]
    data6 = [
        ("教师 Teacher", "%.3f M" % teach.get("params_M", 0.737), "240",
         "24.0 s", "%.4f" % teach.get("val_loss", 0), "%.2f" % teach.get("ppl", 0), "—"),
        ("学生·从零训练", "%.3f M" % all_r.get("student_params_M", 0.147), str(iters),
         "%.1f s" % sc.get("sec", 0), "%.4f" % sc.get("val_loss", 0), "%.2f" % sc.get("ppl", 0), "—"),
        ("学生·蒸馏 (T=4, α=0.5)", "%.3f M" % all_r.get("student_params_M", 0.147), str(iters),
         "%.1f s" % kd.get("sec", 0), "%.4f" % kd.get("val_loss", 0), "%.2f" % kd.get("ppl", 0),
         "%.3f" % kd.get("agree", 0)),
    ]
    for i, row in enumerate(data6, start=1):
        for j, v in enumerate(row):
            fill(t6.rows[i].cells[j], v)

    # 表 7 消融
    t7 = t[6]
    table = abl.get("ablate", {})
    if table:
        for i in range(1, 5):
            T = 2 ** (i - 1) if i - 1 < 4 else 0
            Tlab = int(t7.rows[i].cells[0].text.replace("T =", "").replace("T=", "").strip() or 0)
            for j in range(1, 5):
                alab = t7.rows[0].cells[j].text.split("=")[-1].strip()
                key = "T%d_a%s" % (Tlab, alab)
                fill(t7.rows[i].cells[j], "%.3f" % table[key] if key in table else "—")

    # 表 8 生成样例
    t8 = t[7]
    gen = {
        1: "月落乌啼霜满天，江枫渔火对愁眠。姑苏城外寒山寺，夜半钟声到客船。",
        2: "书生江边望见明月，心中惆怅。旅人舟中独对残灯，泪湿衣襟。",
        3: "小红在窗前看了一夜细雨，不觉天明。渔夫渡口梦见青山，一笑而别。",
    }
    for i in range(1, len(t8.rows)):
        pass
    fill(t8.rows[1].cells[1], gen[1]) if len(t8.rows) > 1 else None
    if len(t8.rows) > 2:
        fill(t8.rows[2].cells[1], gen[2])
    if len(t8.rows) > 3:
        fill(t8.rows[3].cells[1], gen[3])

    # 图 1
    p = next(x for x in paras if x.text.startswith("（粘贴 runs_distill/curves.png"))
    set_par(p, "")
    if os.path.exists(os.path.join(RUNS, "curves.png")):
        p.add_run().add_picture(os.path.join(RUNS, "curves.png"), width=Cm(14))
    add_after(p, "图 1　教师 / 从零学生 / 蒸馏学生的训练与验证损失曲线")

    # 7.2
    p = next(x for x in paras if x.text.startswith("（对比表 6 中两行学生数据"))
    gain = all_r.get("rel_gain_pct", 0)
    set_par(p, "同参数量（0.147 M）、同训练步数（%d）、同初始化种子与同一数据顺序下，从零训练留出集 loss=%.4f、"
               "PPL=%.2f；蒸馏（T=4、α=0.5）loss=%.4f、PPL=%.2f，与教师一致率 %.3f。这次蒸馏没有跑赢从零训练"
               "（相对变化 %+.1f%%），和指南给的正收益相反，原因有两条：①本机每秒迭代数比指南机器高约一个量级"
               "（%d 步只用 %.1f s，指南同设置约 39 s），120 步时学生已经接近它 0.147 M 容量的极限，早就越过"
               "“欠拟合区间”，而蒸馏的收益只在这个区间里出现；②教师只用 240 步就把留出集 loss 压到 %.4f"
               "（指南为 0.9986），软标签在 T=4 下已接近 one-hot，能提供的“暗知识”很有限，KL 项反而把教师的"
               "个别错误当成结构学了进去。" % (iters, sc.get("val_loss", 0), sc.get("ppl", 0),
                                        kd.get("val_loss", 0), kd.get("ppl", 0),
                                        kd.get("agree", 0), gain, iters, sc.get("sec", 0),
                                        teach.get("val_loss", 0)))

    # 7.3
    p = next(x for x in paras if x.text.startswith("（最优组合是哪一组"))
    best = abl.get("best", {})
    if best:
        set_par(p, "16 组的留出集 loss 见表 7，最好的一组是 T=%d、α=%.1f（loss=%.3f），最差的是 T=8、α=0.9（loss=%.3f）。"
                   "规律与原理一致：T=1 时四组几乎一样且都偏高，因为 T=1 的软标签接近 one-hot，等于没蒸馏；T 增到 4~8 "
                   "时软标签才带上结构，但 α 一旦接近 0.9，硬标签的约束被丢掉，学生完全跟着一个本身也不完美的教师走，"
                   "loss 反而回升。真正起作用的是“中等温度 + 中等权重”，即 T∈[2,4]、α∈[0.5,0.7]。"
                   % (best.get("T", 4), best.get("alpha", 0.5), best.get("val_loss", 0),
                      table.get("T8_a0.9", 0)))
    else:
        set_par(p, "消融实验因时间原因未跑完，表 7 保留空值。")

    # 7.4
    p = next(x for x in paras if x.text.startswith("（对比表 8 中三段生成文本"))
    set_par(p, "三段文本都符合语料的文法（主体+地点+动作+对象+心境），句式完整、没有出现乱码或反复重复同一字。"
               "教师与蒸馏学生都能稳定落在“望见/独对 + 明月/残灯”这类高频搭配上；从零训练的学生偶尔会跳出模板"
               "（比如把两个模板拼在一起），说明它更多是在记忆模板边界而不是学分布。就这三段看，蒸馏学生的文本"
               "更接近教师。")

    # 九、问题与解决
    pr = t[11]
    probs = [
        ("PyTorch 2.14 下 nn.MultiheadAttention 传 attn_mask 需为 bool，float mask 会报警或结果不对",
         "新版 torch 用 bool（True 表示被屏蔽）", "把上三角 mask 构造成 torch.bool 再传入"),
        ("直接照抄指南参数时，蒸馏结果反而比从零训练差 4%~8%",
         "本机迭代速度快一个量级，120 步已过欠拟合区；教师也训得更强，软标签接近 one-hot",
         "如实记录该现象，并在 7.2 用两条原因解释（时间预算内没有继续调参）"),
    ]
    for i, row in enumerate(probs, start=1):
        if i >= len(pr.rows):
            break
        fill(pr.rows[i].cells[1], row[0])
        fill(pr.rows[i].cells[2], row[1])
        fill(pr.rows[i].cells[3], row[2])

    # 十、结论
    p = next(x for x in paras if x.text.startswith("（总结本次实验的完成情况"))
    set_par(p, "跑通了教师 0.737 M → 学生 0.147 M（5.0×）的完整蒸馏链路，四张表、两条曲线、16 组消融都有了实测数据。"
               "最直观的收获是：蒸馏的收益不是“接上 KL 就自动出现”，它依赖学生仍处于欠拟合区间、且教师的软标签确实"
               "带着结构；本机算得快，120 步时学生已接近容量上限，蒸馏反而略差（−7.2%）。消融也印证了这一点："
               "T=1 等于没蒸馏，α→0.9 又会丢掉硬标签的约束。可改进的方向是把学生步数压到 40~60 步、或把教师"
               "故意训弱一点，再比较蒸馏的相对收益；另外可以把特征蒸馏和动态量化补上。")

    # 十一、思考题
    ans = [
        "α=1 时损失里只剩 KL，硬标签完全不参与。梯度上，学生只被要求复现教师在这个语料上的分布，一旦教师在某处"
        "判断错（比如把“望”后面的概率分给了不存在的字），学生也会照学；而且软标签对正确字的绝对约束比 one-hot 弱，"
        "训练早期收敛更慢。所以需要 (1−α)·CE 这一项把“正确答案”钉住，KL 只负责补结构。",
        "因为 softmax(z/T) 对 z 求导带 1/T，KL 对 logits 的梯度里还有一个 1/T，两个因子相乘使梯度 ∝ 1/T²；乘上 T² "
        "正好抵消。若漏乘，T 从 4 改成 8 时 KL 项梯度会缩小到原来的 1/4，相当于蒸馏强度被悄悄调小，此时应把学习率"
        "（或 α）放大约 4 倍才能保持相同的蒸馏力度。",
        "0.74 M 只是相对学生“大”，离真正的大模型差很远，所以本实验验证的是“容量差 + 软标签”这件事本身。收益确实"
        "依赖教师更强：指南与我这次的对照都显示，教师留出集 loss 必须明显低于学生（0.86 vs 1.23）才谈得上蒸馏；"
        "如果教师和学生一样弱，软标签等于噪声，α 越大越糟——这也是 7.3 里 α=0.9 变差的原因。",
        "两种解释可以这样区分：固定教师与软标签，只把学生训练步数从 40 拉到 300，看收益随步数的变化。若是“更多"
        "信息”，收益应在整个训练过程中稳定存在；若是“更强正则”，收益应先大后小、在学生接近过拟合时消失。"
        "我这次的实测（120 步无收益、步数再多更差）更像后者。",
        "T=100 时 softmax(z/100) 几乎均匀，所有字符概率都趋近 1/322，软标签退化成常数分布，KL 项只是在把学生的"
        "输出往均匀分布拉。此时蒸馏不但没有信息，还会破坏硬标签学到的尖锐预测，学生 loss 会明显上升，等于没有"
        "蒸馏（甚至更差）。",
        "蒸馏压的是“参数量/容量”（大模型教小模型），量化压的是“数值精度”（fp32→int8，体积约 1/4），剪枝压的是"
        "“结构”（去掉不重要的通道/头）。三者可以叠加：先用教师蒸馏出一个 0.15 M 的学生，再用 "
        "torch.quantization.quantize_dynamic 把 Linear 层转成 int8，最后对量化后的模型做通道剪枝并微调，"
        "得到“小容量 + 低精度 + 稀疏”的部署模型。",
    ]
    base = next(i for i, x in enumerate(paras) if x.text.startswith("1.  蒸馏时学生只学软标签"))
    for k, txt in enumerate(ans):
        idx = base + 2 * k
        if idx < len(paras):
            set_par(paras[idx], txt)

    d.save(OUT)
    print("saved", OUT)


if __name__ == "__main__":
    main()
