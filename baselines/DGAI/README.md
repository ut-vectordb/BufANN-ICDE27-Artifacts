# DGAI: Decoupled On-Disk Graph-Based ANN Index for Efficient Updates and Queries

[![arXiv](https://img.shields.io/badge/arXiv-2510.25401-b31b1b.svg)](https://arxiv.org/abs/2510.25401)
[![DOI](https://img.shields.io/badge/DOI-10.48550/arXiv.2510.25401-blue.svg)](https://doi.org/10.48550/arXiv.2510.25401)

**DGAI** is a novel on-disk graph-based Approximate Nearest Neighbor (ANN) search system designed for large-scale, dynamic vector datasets. 

Unlike traditional coupled architectures, DGAI separates the lightweight graph topology from heavyweight vector data. This decoupled design, combined with a specialized three-stage query strategy and incremental similarity-aware reordering, significantly reduces I/O overhead, delivering state-of-the-art performance for both high-throughput updates and low-latency queries. Experimental results show that the decoupled architecture improves update speed by 8.17x for insertions and 8.16x for deletions, while the three-stage query and incremental reordering enhance query efficiency by 2.57x compared to the traditional coupled architecture.


## 📄 Paper

**DGAI: Decoupled On-Disk Graph-Based ANN Index for Efficient Updates and Queries** *Jiahao Lou, Shufeng Gong, Quan Yu, Hao Guo, Song Yu, Youyou Lu, Yanfeng Zhang, Tiezheng Nie, Ge Yu* arXiv preprint, 2025.

[**Read the Paper**](https://arxiv.org/abs/2510.25401)

## 🔗 Citation
If you find this project or paper useful in your research, please cite our work:
```bibtex
@misc{lou2025dgai,
      title={DGAI: Decoupled On-Disk Graph-Based ANN Index for Efficient Updates and Queries}, 
      author={Jiahao Lou and Shufeng Gong and Quan Yu and Hao Guo and Youyou Lu and Song Yu and Yanfeng Zhang and Tiezheng Nie and Ge Yu},
      year={2026},
      eprint={2510.25401},
      archivePrefix={arXiv},
      primaryClass={cs.DB},
      url={https://arxiv.org/abs/2510.25401}, 
}
```
## 📧 Contact
For any questions, please contact loujh@mails.neu.edu.cn