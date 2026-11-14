# -*- coding: utf-8 -*-
"""自制语言模型的知识蒸馏（纯 CPU，单文件，零外部数据）

用法：
    python distill_lm.py --mode quick      # 30 s 自检
    python distill_lm.py --mode teacher    # 训练教师（约 3~5 min）
    python distill_lm.py --mode all        # 从零 vs 蒸馏
    python distill_lm.py --mode ablate     # T x alpha 消融
"""
import argparse
import json
import math
import os
import random
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

torch.set_num_threads(max(1, min(12, os.cpu_count() or 4)))
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "runs_distill")
os.makedirs(OUT, exist_ok=True)

SUBJ = ["小明", "小红", "阿远", "书生", "渔夫", "牧童", "旅人", "诗人",
        "少年", "故人", "归客", "山僧", "孤客", "游子"]
PLACE = ["江边", "山寺", "长亭", "渡口", "竹林", "荒村", "城楼", "野渡",
         "溪桥", "松间", "篱下", "窗前", "陌上", "舟中"]
VERB = ["望见", "听见", "想起", "送去", "寻到", "梦见", "独对", "遥看",
        "静听", "又逢", "恰遇", "空忆"]
OBJ = ["明月", "落花", "孤舟", "归雁", "残灯", "细雨", "青山", "晚钟",
       "白雪", "寒江", "疏星", "暮云", "秋草", "远帆"]
MOOD = ["心中惆怅", "不觉天明", "欣然归去", "泪湿衣襟", "长叹一声",
        "默然无语", "忽忆旧游", "欲语还休", "此意难平", "一笑而别"]

POEMS = [
    "床前明月光，疑是地上霜。举头望明月，低头思故乡。",
    "春眠不觉晓，处处闻啼鸟。夜来风雨声，花落知多少。",
    "白日依山尽，黄河入海流。欲穷千里目，更上一层楼。",
    "千山鸟飞绝，万径人踪灭。孤舟蓑笠翁，独钓寒江雪。",
    "空山不见人，但闻人语响。返景入深林，复照青苔上。",
    "红豆生南国，春来发几枝。愿君多采撷，此物最相思。",
    "独坐幽篁里，弹琴复长啸。深林人不知，明月来相照。",
    "月落乌啼霜满天，江枫渔火对愁眠。姑苏城外寒山寺，夜半钟声到客船。",
    "两个黄鹂鸣翠柳，一行白鹭上青天。窗含西岭千秋雪，门泊东吴万里船。",
    "朝辞白帝彩云间，千里江陵一日还。两岸猿声啼不住，轻舟已过万重山。",
    "日照香炉生紫烟，遥看瀑布挂前川。飞流直下三千尺，疑是银河落九天。",
    "渭城朝雨浥轻尘，客舍青青柳色新。劝君更尽一杯酒，西出阳关无故人。",
    "寒雨连江夜入吴，平明送客楚山孤。洛阳亲友如相问，一片冰心在玉壶。",
    "独在异乡为异客，每逢佳节倍思亲。遥知兄弟登高处，遍插茱萸少一人。",
    "清明时节雨纷纷，路上行人欲断魂。借问酒家何处有，牧童遥指杏花村。",
    "朱雀桥边野草花，乌衣巷口夕阳斜。旧时王谢堂前燕，飞入寻常百姓家。",
]


def build_corpus(n_lines=3000, seed=0):
    rng = random.Random(seed)
    out = []
    for i in range(n_lines):
        s, p, v, o, m = (rng.choice(x) for x in (SUBJ, PLACE, VERB, OBJ, MOOD))
        if i % 3 == 0:
            out.append("%s%s%s%s，%s。" % (s, p, v, o, m))
        elif i % 3 == 1:
            out.append("%s在%s%s了一夜%s，%s。" % (s, p, o, rng.choice(["看", "听", "守", "忆"]), m))
        else:
            out.append("%s%s道：“%s%s。”%s。" % (s, v, o, rng.choice(["正好", "可怜", "可惜", "依旧"]), m))
        out.append("\n")
    out.extend(POEMS)
    return "".join(out)


class Tokenizer:
    def __init__(self, text):
        chars = sorted(set(text))
        self.stoi = {c: i for i, c in enumerate(chars)}
        self.itos = chars
        self.vocab_size = len(chars)

    def encode(self, s):
        return [self.stoi[c] for c in s if c in self.stoi]


class GPT(nn.Module):
    def __init__(self, vocab, n_layer, n_head, n_embd, block):
        super().__init__()
        self.block = block
        self.tok = nn.Embedding(vocab, n_embd)
        self.pos = nn.Embedding(block, n_embd)
        self.drop = nn.Dropout(0.0)
        self.blocks = nn.ModuleList([
            nn.ModuleDict({
                "ln1": nn.LayerNorm(n_embd),
                "attn": nn.MultiheadAttention(n_embd, n_head, batch_first=True),
                "ln2": nn.LayerNorm(n_embd),
                "mlp": nn.Sequential(nn.Linear(n_embd, 4 * n_embd), nn.GELU(),
                                     nn.Linear(4 * n_embd, n_embd)),
            }) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.head = nn.Linear(n_embd, vocab, bias=False)

    def forward(self, idx, return_hidden=False):
        B, T = idx.shape
        x = self.drop(self.tok(idx) + self.pos(torch.arange(T, device=idx.device)))
        mask = torch.triu(torch.ones(T, T, device=idx.device), diagonal=1).bool()
        for blk in self.blocks:
            h = blk["ln1"](x)
            a, _ = blk["attn"](h, h, h, attn_mask=mask, need_weights=False)
            x = x + a
            x = x + blk["mlp"](blk["ln2"](x))
        x = self.ln_f(x)
        logits = self.head(x)
        if return_hidden:
            return logits, x
        return logits


def n_params(m):
    return sum(p.numel() for p in m.parameters())


def get_batch(data, block, batch, gen):
    ix = torch.randint(len(data) - block - 1, (batch,), generator=gen)
    x = torch.stack([data[i:i + block] for i in ix])
    y = torch.stack([data[i + 1:i + 1 + block] for i in ix])
    return x, y


@torch.no_grad()
def evaluate(model, data, block, batch=24, rounds=20, seed=1234, teacher=None):
    model.eval()
    g = torch.Generator().manual_seed(seed)
    tot, n, agree, ntok = 0.0, 0, 0, 0
    for _ in range(rounds):
        x, y = get_batch(data, block, batch, g)
        logits = model(x)
        tot += F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1)).item()
        n += 1
        if teacher is not None:
            tl = teacher(x)
            agree += (tl.argmax(-1) == logits.argmax(-1)).sum().item()
            ntok += y.numel()
    model.train()
    loss = tot / n
    return loss, math.exp(loss), (agree / ntok if ntok else float("nan"))


def train_student(student, data, block, iters, lr, seed, teacher=None, T=4.0, alpha=0.5, batch=24):
    opt = torch.optim.AdamW(student.parameters(), lr=lr)
    gen = torch.Generator().manual_seed(seed)
    t0 = time.time()
    for it in range(iters):
        x, y = get_batch(data, block, batch, gen)
        logits = student(x)
        ce = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
        if teacher is not None and alpha > 0:
            with torch.no_grad():
                tl = teacher(x)
            kl = F.kl_div(F.log_softmax(logits / T, dim=-1),
                          F.softmax(tl / T, dim=-1), reduction="batchmean") * (T * T)
            loss = (1 - alpha) * ce + alpha * kl
        else:
            loss = ce
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
        opt.step()
    return time.time() - t0


def make_model(cfg):
    return GPT(cfg["vocab"], cfg["n_layer"], cfg["n_head"], cfg["n_embd"], cfg["block"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="all",
                    choices=["quick", "teacher", "all", "ablate"])
    ap.add_argument("--retrain", action="store_true")
    ap.add_argument("--iters", type=int, default=120)
    ap.add_argument("--t-iters", type=int, default=240)
    ap.add_argument("--abl-iters", type=int, default=70)
    ap.add_argument("--block", type=int, default=96)
    ap.add_argument("--batch", type=int, default=24)
    args = ap.parse_args()

    torch.manual_seed(0)
    train_text = build_corpus(3000, seed=0)
    val_text = build_corpus(400, seed=7)
    tok = Tokenizer(train_text + val_text)
    train_data = torch.tensor(tok.encode(train_text), dtype=torch.long)
    val_data = torch.tensor(tok.encode(val_text), dtype=torch.long)
    print("[data] vocab=%d train_chars=%d val_chars=%d block=%d batch=%d"
          % (tok.vocab_size, len(train_data), len(val_data), args.block, args.batch))

    cfg_t = dict(vocab=tok.vocab_size, n_layer=2, n_head=4, n_embd=160, block=args.block)
    cfg_s = dict(vocab=tok.vocab_size, n_layer=2, n_head=4, n_embd=64, block=args.block)

    if args.mode == "quick":
        teacher = make_model(cfg_t)
        student = make_model(cfg_s)
        train_student(teacher, train_data, args.block, 30, 3e-3, seed=1)
        print("[teacher] params = %.3f M" % (n_params(teacher) / 1e6))
        print("[teacher] val_loss=%.4f ppl=%.2f" % evaluate(teacher, val_data, args.block)[:2])
        train_student(student, train_data, args.block, 20, 3e-3, seed=2)
        print("[scratch] val_loss=%.4f ppl=%.2f" % evaluate(student, val_data, args.block)[:2])
        s2 = make_model(cfg_s)
        train_student(s2, train_data, args.block, 20, 3e-3, seed=2, teacher=teacher, T=4, alpha=0.5)
        print("[kd]      val_loss=%.4f ppl=%.2f" % evaluate(s2, val_data, args.block)[:2])
        return

    tpath = os.path.join(OUT, "teacher.pt")
    teacher = make_model(cfg_t)
    if os.path.exists(tpath) and not args.retrain:
        teacher.load_state_dict(torch.load(tpath, map_location="cpu"))
        print("[teacher] loaded", tpath)
    else:
        t0 = time.time()
        train_student(teacher, train_data, args.block, args.t_iters, 3e-3, seed=0)
        dt = time.time() - t0
        torch.save(teacher.state_dict(), tpath)
        print("[teacher] params = %.3f M  train=%.1fs" % (n_params(teacher) / 1e6, dt))
    tl, tp, _ = evaluate(teacher, val_data, args.block)
    print("[teacher] val_loss=%.4f ppl=%.2f" % (tl, tp))
    res = {"teacher": {"params_M": round(n_params(teacher) / 1e6, 3),
                       "val_loss": round(tl, 4), "ppl": round(tp, 2)}}

    if args.mode == "teacher":
        json.dump(res, open(os.path.join(OUT, "results_teacher.json"), "w"), indent=2)
        return

    if args.mode == "all":
        scratch = make_model(cfg_s)
        t_s = train_student(scratch, train_data, args.block, args.iters, 3e-3, seed=7)
        sl, sp, sa = evaluate(scratch, val_data, args.block)
        kd = make_model(cfg_s)
        t_k = train_student(kd, train_data, args.block, args.iters, 3e-3, seed=7,
                            teacher=teacher, T=4.0, alpha=0.5)
        kl, kp, ka = evaluate(kd, val_data, args.block, teacher=teacher)
        print("[student] params = %.3f M  (压缩率 %.1fx)"
              % (n_params(scratch) / 1e6, n_params(teacher) / n_params(scratch)))
        print("[scratch] val_loss=%.4f ppl=%.2f agree=%.3f %.1fs" % (sl, sp, sa, t_s))
        print("[kd]      val_loss=%.4f ppl=%.2f agree=%.3f %.1fs" % (kl, kp, ka, t_k))
        print("[对比] 从零 %.4f -> 蒸馏 %.4f  相对下降 %.1f%%" % (sl, kl, (sl - kl) / sl * 100))
        res.update({"student_params_M": round(n_params(scratch) / 1e6, 3),
                    "scratch": {"val_loss": round(sl, 4), "ppl": round(sp, 2),
                                "agree": round(sa, 3), "sec": round(t_s, 1)},
                    "kd": {"val_loss": round(kl, 4), "ppl": round(kp, 2),
                           "agree": round(ka, 3), "sec": round(t_k, 1), "T": 4.0, "alpha": 0.5},
                    "rel_gain_pct": round((sl - kl) / sl * 100, 1),
                    "iters": args.iters})
        json.dump(res, open(os.path.join(OUT, "results_all.json"), "w"), indent=2, ensure_ascii=False)
        return

    if args.mode == "ablate":
        # 先缓存 32 个批次的教师 logits，避免每组重复前向
        cache = []
        gen = torch.Generator().manual_seed(99)
        with torch.no_grad():
            for _ in range(32):
                x, y = get_batch(train_data, args.block, args.batch, gen)
                cache.append((x, y, teacher(x)))
        table = {}
        best = (1e9, None)
        for T in (1, 2, 4, 8):
            for a in (0.3, 0.5, 0.7, 0.9):
                st = make_model(cfg_s)
                opt = torch.optim.AdamW(st.parameters(), lr=3e-3)
                st.train()
                for it in range(args.abl_iters):
                    x, y, tl = cache[it % len(cache)]
                    logits = st(x)
                    ce = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
                    kl = F.kl_div(F.log_softmax(logits / T, dim=-1),
                                  F.softmax(tl / T, dim=-1), reduction="batchmean") * (T * T)
                    loss = (1 - a) * ce + a * kl
                    opt.zero_grad(set_to_none=True)
                    loss.backward()
                    opt.step()
                vl = evaluate(st, val_data, args.block)[0]
                table["T%d_a%.1f" % (T, a)] = round(vl, 4)
                print("[ablate] T=%d alpha=%.1f val_loss=%.4f" % (T, a, vl))
                if vl < best[0]:
                    best = (vl, (T, a))
        print("[ablate] best: T=%d alpha=%.1f val_loss=%.4f" % (best[1][0], best[1][1], best[0]))
        res = json.load(open(os.path.join(OUT, "results_all.json"))) \
            if os.path.exists(os.path.join(OUT, "results_all.json")) else {}
        res["ablate"] = table
        res["best"] = {"T": best[1][0], "alpha": best[1][1], "val_loss": round(best[0], 4)}
        json.dump(res, open(os.path.join(OUT, "results_ablate.json"), "w"), indent=2, ensure_ascii=False)
        return


if __name__ == "__main__":
    main()
