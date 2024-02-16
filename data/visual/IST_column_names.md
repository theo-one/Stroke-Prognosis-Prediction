1. randhosp_id  医院标识符
2. randpat_id 患者标识符
3. pretrialexp 实验前的经验或者条件
4. country 国家
5. trialphase 临床实验的阶段
6. phase 
7. <mark>itt_treat 治疗方式（安慰剂或tr-PA）</mark>
8. <mark>age  年龄 </mark>
9. <mark>gender 性别</mark>
10. deathcode 参与者死亡的相关编码
11. deathdate_unknown 死亡日期是否未知
12. randyear 年份
13. randmonth 月份
14. randhour 小时
15. randmin 分钟
16. randdelay 
17. livealone_rand 是否独居
18. indepinadl_rand 在日常生活活动中的独立性
19. nobleed_rand 是否无出血
20. infarct 最近的缺血性改变可能是这次中风的原因吗
21. antiplat_rand 使用抗血小板药物
22. <mark>atrialfib_rand 是否有心房颤动</mark>
23. sbprand 收缩压，可以用来判断高血压
24. dbprand 舒张压
25. weight 体重
26. glucose 血糖水平
+ 格拉斯哥昏迷评分
1.  gcs_eye_rand 眼睛反应
2.  gcs_motor_rand 运动反应
3.  gcs_verbal_rand 言语反应
4.  <mark>gcs_score_rand 分数
5.  <mark>nihss 国立卫生研究院卒中量表（NIH Stroke Scale）得分，用于评估卒中的严重程度
6.  liftarms_rand 能否举起手臂
7.  ablewalk_rand 能够行走
8.  weakface_rand 面部
9.  weakarm_rand 手臂
10. weakleg_rand 腿部的虚弱情况
11. dysphasia_rand 言语障碍（语言理解或表达困难）
12. hemianopia_rand 半盲（视野丧失一半的情况）
13. visuospat_rand 视觉空间障碍情况
14. brainstemsigns_rand 脑干症状的存在
15. otherdeficit_rand 随机化时其他神经功能缺失
16. <mark>stroketype 中风的类型（如缺血性或出血性）
17. pred_nihss 预测的国立卫生研究院卒中量表（NIH Stroke Scale）得分
18. konprob 可能指的是某种特定的概率或风险评估
19. randvioltype 可能指随机化过程中的违规类型
20. other_antiplat_pre 随机化前使用的其他抗血小板药物
21. anticoag_pre 随机化前使用的抗凝血药物
22. recinfus 可能指接受输液的情况
23. gotbolus 是否接受了冲击剂量的药物（常用于急救或治疗开始阶段）
24. infus_start 输液开始的时间
25. infus_halt 输液中止的时间
26. totdose 总剂量
+ 分别表示开始时、30分钟、60分钟和24小时后的收缩压
1.  sbpstart 
2.  sbp30min
3.  sbp60min
4.  sbp24h
+ 分别表示开始时、30分钟、60分钟和24小时后的舒张压
1.  dbpstart
2.  dbp30min
3.  dbp60min
4.  dbp24h
5.  <mark>treatdelay 治疗延迟的时间
6.  recR R扫描
7.  asl 病变的判定侧
8.  R_scantype R扫描
9.  R_scannorm R扫描的结果是否正常
10. recP 是否接受P扫描
11. P_scantype P扫描的类型
12. P_scannorm P扫描的类型是否正常
13. rec7 是否收到7天表格
14. consent_type 参与者同意参加试验的类型或方式
15. aspirin_pre 入院前是否服用阿司匹林
16. dipyridamole_pre  入院前是否服用双嘧达莫
17. clopidogrel_pre 入院前是否服用氯毗格雷
18. lowdose_heparin_pre 入院前低剂量肝素
19. fulldose_heparin_pre 入院前全剂量肝素
20. warfarin_pre 入院前服用华法林
21. antithromb_pre  入院前还有其他抗血栓药物吗
22. <mark>hypertension_pre 入院前是否有高血压治疗
23. <mark>diabetes_pre 入院前是否接受过糖尿病治疗
24. <mark>stroke_pre 是否有过中风或TIA（短暂性脑缺血发作）病史
25. aspirin_day1 第一天服用阿司匹林了吗？
26. antiplatelet_day1 第一天还有其他抗血小板药物吗？
27. lowdose_heparin_day1 低剂量肝素第一天
28. full_anticoag_day1 24小时内全抗凝吗
29. lowerBP_day1 24小时内降低血压的治疗方法
30. nontrial_thromb_day1 24小时内进行非试验性溶栓吗？
31. iv_fluids_day1 24小时内静脉注射吗
32. insulin_day1 24小时注射胰岛素吗
33. aspirin_days2to7 
34. antiplatelet_days2to7
35. lowdose_heparin_days2to7
36. full_anticoag_days2to7
37. lowerBP_days2to7
38. nontrial_thromb_days2to7
39. nasogastric_days2to7 鼻胃管喂养第二-七天
40. antibiotics_days2to7 抗生素治疗第二-七天
41. findiag7 试验第七天的最终诊断
42.  <mark>brainsite7 初始缺血性脑卒中的位置
43.  haem_type7 最初出血性中风的类型
44.  nonstroke_type7 非中风原因类型
45.  med_adno 前七天的住院天数
46.  critcareno 前七天在重症监护病房的夜数
47.   strk_unitno 前七天中风病房的住院天数
48.  genwardno 前七天在普通病房住了多少个晚上
49.  adjudicated 患者在前七天内判定事件
50.  sevendaycase 前七天的裁判项目类型
51.  final_status 最终裁定状态
52.  sich7 7天内出现症状性颅内出血
53.  dead7 在7天内是否死亡
54.  myocard_infarct 7天内心肌梗死
55.  extracranial_bleed 7天内颅内出血严重
56.  allergic_reaction 7天内有严重过敏反应
57.  other_effect 7天内有其他副作用
58.  adverse_reaction 7天内有其他不良反应
59.  gcs_eye_7  7天内最佳视力反应
60.  gcs_motor_7 7天内最佳运动反应
61.  gcs_verbal_7 7天内最佳言语反应
62.  liftarms_7 第七天是否能够举起手臂
63.  ablewalk_7 第七天是否能够独立行走
64.  indepinadl_7 第七天患者在日常活动中的独立性
65.  destination 患者在出院或者评估后的去向
66.  recsix 收到6个月表格
67.  sixmonthform 6个月表格类型
68.  sixcompleted_by 填写6个月表格的人
69.  <mark>ohs6 6个月的OHS表格
70.  ordinal6 6个月的职业健康安全表，包括4 5 6级
71.  aliveind6 6个月存活并独立
72.  alivefav6 6个月生存率和预后良好
73.  dead6mo 六个月内患者是否死亡
74.  imputed6 6个月前的OHS表
75.  aspirin6 入院当天服用阿司匹林
76.  bloodthin6 入院当天注射血液稀释剂
77.  clotbust6 入院当天服用了抗凝血药物
78.  stocking6 入院当天是否使用抗栓袜
79.  gotprobs6 中风给患者留下很多问题吗
80.  needhelp6 日常活动需要帮助吗
81.  walkhelp6 走路需要帮助吗
82.  speakprob6 是否有说话问题
83.  mobility6 行动能力
84.  selfcare6 自我照顾能力
85.  activities6 日常活动能力
86.  pain6 疼痛或不适
87.  anxiety6 焦虑或抑郁
88.  wherelive6 居住地点
89.  howlive6 生活情况
90.  euroqol6 6个月时的健康状况
91.  receighteen 十八个月后的回访或评估记录
92.  eighteenmonthform 18个月表格类型
93.  eighteencompleted_by 填写18个月表格的人
94.  aspirin18 入院当天给予阿司匹林
95.  bloodthin18  入院当天进行血液稀释剂
96.  clotbust18 入院当天服用的抗凝血药物
97.  stocking18 入院当天抗血栓袜子
98.  gotprobs18 中风患者遗留问题
99.  needhelp18 日常活动是否需要帮助
100. walkhelp18 是否需要行走辅助
101. speakprob18 是否有言语问题
102. mobility18 十八个月患者行动能力
103. selfcare18 自我照顾能力
104. activities18 日常活动能力
105. pain18 疼痛或不适
106. anxiety18 焦虑或抑郁
107. wherelive18 患者的居住地点
108. howlive18 居住方式
109. euroqol18 18个月是的健康状况
110. deadordep18 是否死亡或依赖他人
111. ohs18 健康评分
112. ordinal18 18个月的职业健康安全表，等级4 5 6加起来
113. aliveind18 是否独立生活
114. alivefav18 18个月后生存率和预后良好
115. yrfu_code 缺少18个月跟踪的原因
116. waiver_code 放弃同意的理由
117. extracranial_bleed_site 颅内出血部位
118. other_effect_code 记录其他效应或事件的代码
119. nostartcode 治疗或试验未能开始的原因相关的代码
120. deadordep6 可能与六个月时患者的生存或者依赖状态相关
121. missing6 缺失数据
122. missing18 
123. event_days 特定事件发生后经过的天数
124. agecat 年龄分类
125. ageimp 年龄的估算或推断
126. age_true 实际的或准备的年龄
127. R_acuteic R扫描显示急性缺血性改变？
128. hypodeg 急性低密度的程度
129. hypodegsite 最大的可见肿胀部位
130. mca 大脑中动脉的改变
131. affmca 中脑区病变部位/大小
132. aspcau 尾状核区缺血改变
133. asplen 透镜状区显示缺血改变
134. aspins 脑岛区出现缺血改变
135. aspint 内囊区显示缺血改变
136. aspm1  M1区缺血改变
137. aspm2  M2区缺血改变
138. aspm3 
139. aspm4 
140. aspm5 
141. aspm6 
142. oial 脑的其他部位与急性缺血性损伤有关
143. aca 大脑前动脉受影响
144. pca 大脑后大脑动脉受影响
145. subinf 急性皮质下小梗死
146. cbzinf 急性皮质交界区梗死
147. cinf 急性小脑梗塞
148. stem 急性脑干梗塞
149. tisswell 急性梗死时组织肿胀程度
150. hdart1 中动脉主干呈高密度征象
151. hdart2 岛状中动脉呈高密度征象
152. hdart3 颈内动脉呈高密度征象
153. hdart4 大脑前动脉呈高密度征象
154. hdart5 大脑后动脉呈高密度征象
155. hdart6 基底动脉呈高密度征象
156. hdart7 椎动脉呈高密度征象
157. seclesion 另一个小的新缺血性损伤
158. redbtvola 中央萎缩
159. redbtvolb 大脑皮层萎缩
160. wmla 前白质透明
161. wmlp 后白质透明
162. oldlesion 有旧的血管病变吗
163. oldlesion1 老年性皮质梗塞
164. oldlesion2 陈旧性纹状囊性梗死
165. oldlesion3 旧边界区梗死
166. oldlesion4 旧的胫隙性梗死
167. oldlesion5 老年性脑干/小脑梗死
168. oldlesion6 老年性出血
169. nslesion 非特异性脑病变的存在
170. nslesion1 大脑肿瘤
171. nslesion2 脑炎
172. nslesion3 脱髓鞘
173. nslesion4 大脑脓肿
174. nslesion5 其他
175. scanqual 扫描质量
176. R_infarct_size R扫描的梗死面积
177. apca ACA/PCA区域梗死
178. subinf1 急性皮质下小梗死
179. cbzinf1 急性皮质交界区梗死
180. cinf1 急性小脑梗塞
181. stem1 急性脑干梗塞
182. aca1 大脑前动脉受影响
183. pca1 大脑后动脉受影响
184. R_infarct_territory 脑梗死的领域
185. R_hypodensity 急性低密度程度
186. R_swelling 急性梗死时组织肿胀程度
187. R_hyperdense_arteries 可见高密度动脉
188. R_atrophy R扫描可见萎缩
189. R_whitematter R扫描可见白质
190. R_oldlesion 旧血管病变
191. R_nonstroke_lesion 非脑卒中病变
192. R_isch_change 可见缺血改变
193. R_mca_aspects 大脑中动脉各方面评分
194. R_tot_aspects 总分
195. R_hyperdensity 右侧脑部的高密度区域
196. vis_infarct 可见的脑梗死
197. time_to_PH 到达肺高压的时间
198. PH_haemorrhage 与肺高压相关的出血
199. surv6 
200. censor6
201. censor18
202. surv18
203. plan18
204. UKextra
205. disab_unknown6 六个月患者的残疾状态
206. vital_and_disabunknown6  患者的生命体征在六个月
207. disab_unknown18 十八个月的残疾状态是否未知
208. vital_and_disabunknown18 
209. treatment  患者接受的特定治疗或治疗方案
210. haltcode 停止治疗或实验的特定代码或原因
