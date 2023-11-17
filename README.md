# Quantitative three-dimensional imaging of chemical short-range order via machine learning enhanced atom probe tomography

Published in Nature Communications
DOI:
https://doi.org/10.1038/s41467-023-43314-y

![image](https://user-images.githubusercontent.com/44220131/223120821-8e69b316-724b-40e0-9c84-ee76207757b5.png)

_Yue Li1,*, Ye Wei1, Zhangwei Wang2,*, Xiaochun Liu3, Timoteo Colnaghi4, Liuliu Han1, Ziyuan Rao1, Xuyang Zhou1, Liam Huber1, Raynol Dsouza1, Yilun Gong1, Jörg Neugebauer1, Andreas Marek4, Markus Rampp4, Stefan Bauer5, Hongxiang Li6, Ian Baker7, Leigh T. Stephenson1, Baptiste Gault1, 8,*_

1 Max-Planck Institut für Eisenforschung GmbH, Max-Planck-Straße 1, 40237 Düsseldorf, Germany

2 State Key Laboratory of Powder Metallurgy, Central South University, Changsha, 410083, China

3 Institute of Metals, College of Materials Science and Engineering, Changsha University Of Science and Technology, Changsha 410114, China

4 Max Planck Computing and Data Facility, Gießenbachstraße 2, 85748 Garching, Germany

5 Max Planck Institute for Intelligent Systems, Max-Planck-Ring 4, 72076 Tübingen, Germany

6 State Key Laboratory for Advanced Metals and Materials, University of Science and Technology Beijing, 100083, Beijing, China

7 Thayer School of Engineering, 14 Engineering Drive, Dartmouth College, Hanover, NH 03755, USA

8 Department of Materials, Imperial College, South Kensington, London SW7 2AZ, UK

*Corresponding authors, yue.li@mpie.de (Y. L.); z.wang@csu.edu.cn (Z. W.); b.gault@mpie.de (B. G.)

Chemical short-range order (CSRO) refers to atoms of specific elements self-organising within a disordered crystalline matrix to form particular atomic neighbourhoods. CSRO is typically characterized indirectly, using volume-averaged or through projection microscopy techniques that fail to capture the three-dimensional atomistic architectures. Here, we present a machine-learning enhanced approach to break the inherent resolution limits of atom probe tomography enabling three-dimensional imaging of multiple CSROs. We showcase our approach by addressing a long-standing question encountered in body-centred-cubic Fe-Al alloys that see anomalous property changes upon heat treatment. We use it to evidence non-statistical B2-CSRO instead of the generally-expected D03-CSRO. We introduce quantitative correlations among annealing temperature, CSRO, and nano-hardness and electrical resistivity. Our approach is further validated on modified D03-CSRO detected in Fe-Ga. The proposed strategy can be generally employed to investigate short/medium/long-range ordering phenomena in different materials and help design future high-performance materials.

The codes incude three modules. 1 Synthetic z-SDMs bank: Generating artificial APT data along the <002> containing either a randomly distributed BCC-matrix, D03 and B2 CSRO. 2 1DCNN: Training 1DCNN to obtain an BCC-matrix/D03-CSRO/B2-CSRO multi-class classification model. 3 Exp: Applications of this model in FeAl to obtain the 3D CSRO distributions. 

System requirements:
The CNN was implemented using Keras 2.2.4 with the TensorFlow 1.13.1 backend on Python 3.7. Others are performed on Python 3.7.

Demo data:
An annealed Fe-Al APT demo data is provided in https://doi.org/10.6084/m9.figshare.23989050.
