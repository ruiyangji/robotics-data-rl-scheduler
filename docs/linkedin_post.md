# LinkedIn & Social Media Announcement Post

*Copy-paste ready for LinkedIn, X/Twitter, or personal website publication.*

---

### LinkedIn Post Draft

When I was at Meta this summer working in Capacity Efficiency, I built the Chronos evaluation pipeline to backtest an internal AI on 2,000+ historical datacenter capacity proposals. The biggest lesson from that work wasn't about the model—it was about **evaluation rigor and data infrastructure**.

When I returned to Cornell, I had a working discrete-event cluster scheduler using PPO in PyTorch to pack compute jobs under CPU and memory constraints. 

Around the same time, I started looking into robotics infrastructure. In robotics, policy learning gets all the attention. But the first wall you hit in production isn't the policy architecture—it’s **messy data infrastructure**:
- Multi-rate streams: high-frequency joint states at 50–100Hz, camera metadata at 10–30Hz, heartbeats at 1–5Hz.
- Physical network chaos: wireless jitter, packet drops, and hardware clock skews.
- Bandwidth bottlenecks: fleets competing for edge gateway bandwidth with strict freshness deadlines.

When you look at that list through a systems lens, it looked familiar: **it’s a distributed scheduling and streaming problem.**

I haven’t worked on robotics data professionally, but the infrastructure problems looked like the systems challenges I love. So I built an open-source systems artifact: **pointing my RL cluster scheduler at a multi-rate robotics telemetry pipeline.**

Here is what I built:
1. **Multi-Rate Ingest & Fault Injection**: Simulates an 8–32 robot fleet with latency jitter, 5% loss, and ±80ms clock drift.
2. **Clock Synchronization & Watermarking**: An asymmetric transit filter corrects clock skew in real time, with stream watermarks $W(t)$ defining completeness frontiers.
3. **Stale Packet Policy**: Real-time control needs fresh data; joint packets arriving older than 500ms are dropped with explicit root-cause attribution.
4. **Action-Masked PPO**: Hard-enforcing physical bandwidth and battery constraints inside the categorical policy accelerated training convergence by **+40%** and completely eliminated invalid action violations.
5. **Permutation-Invariant Set-Attention**: By using Transformer self-attention without positional encodings, the policy treats robots as permutation-invariant sets—achieving **zero-shot generalization to clusters 2x larger** (10 nodes ➔ 20 nodes) without retraining.
6. **The Chronos Habit**: Replaying identical traces against deterministic baselines (FIFO, Round Robin, Shortest Job First, Highest Value First) and building an automated failure analysis harness that diagnoses exactly *where and why* PPO lost to baselines.

The part I cared about most wasn't whether PPO won. It was building the harness that could tell me honestly when it lost to a simple heuristic like FIFO, and why (e.g. greedy heuristics outperforming RL during bursty arrival spikes).

Check out the full repo, architecture diagrams, interactive Streamlit replay dashboard, and technical write-up below!

🔗 **GitHub**: https://github.com/ruiyangji/robotics-data-rl-scheduler
📝 **Full Write-Up**: https://github.com/ruiyangji/robotics-data-rl-scheduler/blob/main/docs/blog_post.md

Huge thanks to everyone who provided feedback on the evaluation architecture! Would love to hear thoughts from engineers working on distributed systems and robotics platforms.

#DistributedSystems #ReinforcementLearning #PyTorch #Robotics #DataInfrastructure #SystemsEngineering #MachineLearning
