"""Synthetic text-image rendering for DualTSR pretraining.

渲染流程仿照 SynthText 风格的文本行合成:从语料/字符集采样文本,随机选
字体、颜色、背景与几何扰动,渲染出 HR 文本行图像。LR 仍由训练时的在线
blind degradation 生成(见 data.degrade_tensor),与 CTR-TSR 保持一致。
"""

from __future__ import annotations

import math
import random
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont
from torch.utils.data import Dataset

FONT_SUFFIXES = {".ttf", ".otf", ".ttc"}

# 字符集允许的 Unicode 区间:ASCII 可见字符、CJK 标点、全角符号、CJK 统一汉字。
CHARSET_RANGES: tuple[tuple[int, int], ...] = (
    (0x21, 0x7E),
    (0x3001, 0x3011),
    (0xFF01, 0xFF1F),
    (0x4E00, 0x9FFF),
)

DEFAULT_RENDER_CFG: dict[str, Any] = {
    "font_size_min": 64,
    "font_size_max": 112,
    "margin_ratio_min": 0.10,
    "margin_ratio_max": 0.45,
    "spacing_ratio_min": -0.04,
    "spacing_ratio_max": 0.30,
    "baseline_jitter_ratio": 0.04,
    "rotation_max_deg": 3.0,
    "perspective_jitter_ratio": 0.05,
    "perspective_prob": 0.4,
    "min_aspect_ratio": 2.2,
    "max_aspect_ratio": 5.0,
    "contrast_min": 60,
    "stroke_prob": 0.25,
    "stroke_width_max": 3,
    "shadow_prob": 0.25,
    "background_weights": {"solid": 0.35, "gradient": 0.3, "noise": 0.25, "image": 0.1},
    "noise_sigma_min": 4.0,
    "noise_sigma_max": 28.0,
    "final_blur_prob": 0.15,
    "final_blur_max": 0.6,
}

DEFAULT_TEXT_CFG: dict[str, Any] = {
    "corpus_prob": 0.7,
    "mode_weights": {"chinese": 0.55, "mixed": 0.2, "alnum": 0.15, "digits": 0.1},
    "mean_length": 6.0,
    "common_ratio": 0.80,
    "less_common_ratio": 0.10,
}

# ---------------------------------------------------------------------------
# 现代汉语常用字表 (GB, 1988)
# 前 2500 字 = 一级(常用字), 后 1000 字 = 二级(次常用字).
# 组内按笔画排序(非频率序), 但 *组间* 本身就是频率分层.
# 来源: https://gist.github.com/jjgod/1432945 (公共领域)
# ---------------------------------------------------------------------------
_COMMON_2500 = (
    "一乙二十丁厂七卜人入八九几儿了力乃刀又三于干亏士工土才寸下大丈与万上小口巾山千乞川亿个勺久凡及夕丸么"
    "广亡门义之尸弓己已子卫也女飞刃习叉马乡丰王井开夫天无元专云扎艺木五支厅不太犬区历尤友匹车巨牙屯比互切"
    "瓦止少日中冈贝内水见午牛手毛气升长仁什片仆化仇币仍仅斤爪反介父从今凶分乏公仓月氏勿欠风丹匀乌凤勾文"
    "六方火为斗忆订计户认心尺引丑巴孔队办以允予劝双书幻玉刊示末未击打巧正扑扒功扔去甘世古节本术可丙左厉右"
    "石布龙平灭轧东卡北占业旧帅归且旦目叶甲申叮电号田由史只央兄叼叫另叨叹四生失禾丘付仗代仙们仪白仔他斥瓜"
    "乎丛令用甩印乐句匆册犯外处冬鸟务包饥主市立闪兰半汁汇头汉宁穴它讨写让礼训必议讯记永司尼民出辽奶奴加召"
    "皮边发孕圣对台矛纠母幼丝式刑动扛寺吉扣考托老执巩圾扩扫地扬场耳共芒亚芝朽朴机权过臣再协西压厌在有百存"
    "而页匠夸夺灰达列死成夹轨邪划迈毕至此贞师尘尖劣光当早吐吓虫曲团同吊吃因吸吗屿帆岁回岂刚则肉网年朱先丢"
    "舌竹迁乔伟传乒乓休伍伏优伐延件任伤价份华仰仿伙伪自血向似后行舟全会杀合兆企众爷伞创肌朵杂危旬旨负各名"
    "多争色壮冲冰庄庆亦刘齐交次衣产决充妄闭问闯羊并关米灯州汗污江池汤忙兴宇守宅字安讲军许论农讽设访寻那迅"
    "尽导异孙阵阳收阶阴防奸如妇好她妈戏羽观欢买红纤级约纪驰巡寿弄麦形进戒吞远违运扶抚坛技坏扰拒找批扯址走"
    "抄坝贡攻赤折抓扮抢孝均抛投坟抗坑坊抖护壳志扭块声把报却劫芽花芹芬苍芳严芦劳克苏杆杠杜材村杏极李杨求更"
    "束豆两丽医辰励否还歼来连步坚旱盯呈时吴助县里呆园旷围呀吨足邮男困吵串员听吩吹呜吧吼别岗帐财针钉告我乱"
    "利秃秀私每兵估体何但伸作伯伶佣低你住位伴身皂佛近彻役返余希坐谷妥含邻岔肝肚肠龟免狂犹角删条卵岛迎饭饮"
    "系言冻状亩况床库疗应冷这序辛弃冶忘闲间闷判灶灿弟汪沙汽沃泛沟没沈沉怀忧快完宋宏牢究穷灾良证启评补初社"
    "识诉诊词译君灵即层尿尾迟局改张忌际陆阿陈阻附妙妖妨努忍劲鸡驱纯纱纳纲驳纵纷纸纹纺驴纽奉玩环武青责现表"
    "规抹拢拔拣担坦押抽拐拖拍者顶拆拥抵拘势抱垃拉拦拌幸招坡披拨择抬其取苦若茂苹苗英范直茄茎茅林枝杯柜析板"
    "松枪构杰述枕丧或画卧事刺枣雨卖矿码厕奔奇奋态欧垄妻轰顷转斩轮软到非叔肯齿些虎虏肾贤尚旺具果味昆国昌"
    "畅明易昂典固忠咐呼鸣咏呢岸岩帖罗帜岭凯败贩购图钓制知垂牧物乖刮秆和季委佳侍供使例版侄侦侧凭侨佩货依"
    "的迫质欣征往爬彼径所舍金命斧爸采受乳贪念贫肤肺肢肿胀朋股肥服胁周昏鱼兔狐忽狗备饰饱饲变京享店夜庙府底"
    "剂郊废净盲放刻育闸闹郑券卷单炒炊炕炎炉沫浅法泄河沾泪油泊沿泡注泻泳泥沸波泼泽治怖性怕怜怪学宝宗定宜审"
    "宙官空帘实试郎诗肩房诚衬衫视话诞询该详建肃录隶居届刷屈弦承孟孤陕降限妹姑姐姓始驾参艰线练组细驶织终驻驼"
    "绍经贯奏春帮珍玻毒型挂封持项垮挎城挠政赴赵挡挺括拴拾挑指垫挣挤拼挖按挥挪某甚革荐巷带草茧茶荒茫荡荣故"
    "胡南药标枯柄栋相查柏柳柱柿栏树要咸威歪研砖厘厚砌砍面耐耍牵残殃轻鸦皆背战点临览竖省削尝是盼眨哄显哑冒"
    "映星昨畏趴胃贵界虹虾蚁思蚂虽品咽骂哗咱响哈咬咳哪炭峡罚贱贴骨钞钟钢钥钩卸缸拜看矩怎牲选适秒香种秋科重"
    "复竿段便俩贷顺修保促侮俭俗俘信皇泉鬼侵追俊盾待律很须叙剑逃食盆胆胜胞胖脉勉狭狮独狡狱狠贸怨急饶蚀饺"
    "饼弯将奖哀亭亮度迹庭疮疯疫疤姿亲音帝施闻阀阁差养美姜叛送类迷前首逆总炼炸炮烂剃洁洪洒浇浊洞测洗活派洽"
    "染济洋洲浑浓津恒恢恰恼恨举觉宣室宫宪突穿窃客冠语扁袄祖神祝误诱说诵垦退既屋昼费陡眉孩除险院娃姥姨姻娇"
    "怒架贺盈勇怠柔垒绑绒结绕骄绘给络骆绝绞统耕耗艳泰珠班素蚕顽盏匪捞栽捕振载赶起盐捎捏埋捉捆捐损都哲逝"
    "捡换挽热恐壶挨耻耽恭莲莫荷获晋恶真框桂档桐株桥桃格校核样根索哥速逗栗配翅辱唇夏础破原套逐烈殊顾轿较顿"
    "毙致柴桌虑监紧党晒眠晓鸭晃晌晕蚊哨哭恩唤啊唉罢峰圆贼贿钱钳钻铁铃铅缺氧特牺造乘敌秤租积秧秩称秘透笔笑"
    "笋债借值倚倾倒倘俱倡候俯倍倦健臭射躬息徒徐舰舱般航途拿爹爱颂翁脆脂胸胳脏胶脑狸狼逢留皱饿恋桨浆衰高席"
    "准座脊症病疾疼疲效离唐资凉站剖竞部旁旅畜阅羞瓶拳粉料益兼烤烘烦烧烛烟递涛浙涝酒涉消浩海涂浴浮流润浪浸"
    "涨烫涌悟悄悔悦害宽家宵宴宾窄容宰案请朗诸读扇袜袖袍被祥课谁调冤谅谈谊剥恳展剧屑弱陵陶陷陪娱娘通能难预"
    "桑绢绣验继球理捧堵描域掩捷排掉堆推掀授教掏掠培接控探据掘职基著勒黄萌萝菌菜萄菊萍菠营械梦梢梅检梳梯桶救"
    "副票戚爽聋袭盛雪辅辆虚雀堂常匙晨睁眯眼悬野啦晚啄距跃略蛇累唱患唯崖崭崇圈铜铲银甜梨犁移笨笼笛符第敏做"
    "袋悠偿偶偷您售停偏假得衔盘船斜盒鸽悉欲彩领脚脖脸脱象够猜猪猎猫猛馅馆凑减毫麻痒痕廊康庸鹿盗章竟商族旋"
    "望率着盖粘粗粒断剪兽清添淋淹渠渐混渔淘液淡深婆梁渗情惜惭悼惧惕惊惨惯寇寄宿窑密谋谎祸谜逮敢屠弹随蛋隆"
    "隐婚婶颈绩绪续骑绳维绵绸绿琴斑替款堪搭塔越趁趋超提堤博揭喜插揪搜煮援裁搁搂搅握揉斯期欺联散惹葬葛董葡"
    "敬葱落朝辜葵棒棋植森椅椒棵棍棉棚棕惠惑逼厨厦硬确雁殖裂雄暂雅辈悲紫辉敞赏掌晴暑最量喷晶喇遇喊景践跌跑"
    "遗蛙蛛蜓喝喂喘喉幅帽赌赔黑铸铺链销锁锄锅锈锋锐短智毯鹅剩稍程稀税筐等筑策筛筒答筋筝傲傅牌堡集焦傍储奥"
    "街惩御循艇舒番释禽腊脾腔鲁猾猴然馋装蛮就痛童阔善羡普粪尊道曾焰港湖渣湿温渴滑湾渡游滋溉愤慌惰愧愉慨割"
    "寒富窜窝窗遍裕裤裙谢谣谦属屡强粥疏隔隙絮嫂登缎缓编骗缘瑞魂肆摄摸填搏塌鼓摆携搬摇搞塘摊蒜勤鹊蓝墓幕"
    "蓬蓄蒙蒸献禁楚想槐榆楼概赖酬感碍碑碎碰碗碌雷零雾雹输督龄鉴睛睡睬鄙愚暖盟歇暗照跨跳跪路跟遣蛾蜂嗓置罪"
    "罩错锡锣锤锦键锯矮辞稠愁筹签简毁舅鼠催傻像躲微愈遥腰腥腹腾腿触解酱痰廉新韵意粮数煎塑慈煤煌满漠源滤滥"
    "滔溪溜滚滨粱滩慎誉塞谨福群殿辟障嫌嫁叠缝缠静碧璃墙撇嘉摧截誓境摘摔聚蔽慕暮蔑模榴榜榨歌遭酷酿酸磁愿需"
    "弊裳颗嗽蜻蜡蝇蜘赚锹锻舞稳算箩管僚鼻魄貌膜膊膀鲜疑馒裹敲豪膏遮腐瘦辣竭端旗精歉熄熔漆漂漫滴演漏慢寨赛"
    "察蜜谱嫩翠熊凳骡缩慧撕撒趣趟撑播撞撤增聪鞋蕉蔬横槽樱橡飘醋醉震霉瞒题暴瞎影踢踏踩踪蝶蝴嘱墨镇靠稻黎"
    "稿稼箱箭篇僵躺僻德艘膝膛熟摩颜毅糊遵潜潮懂额慰劈操燕薯薪薄颠橘整融醒餐嘴蹄器赠默镜赞篮邀衡膨雕磨凝辨"
    "辩糖糕燃澡激懒壁避缴戴擦鞠藏霜霞瞧蹈螺穗繁辫赢糟糠燥臂翼骤鞭覆蹦镰翻鹰警攀蹲颤瓣爆疆壤耀躁嚼嚷籍魔"
    "灌蠢霸露囊罐"
)
_LESS_COMMON_1000 = (
    "匕刁丐歹戈夭仑讥冗邓艾夯凸卢叭叽皿凹囚矢乍尔冯玄邦迂邢芋芍吏夷吁吕吆屹廷迄臼仲伦伊肋旭匈凫妆亥汛"
    "讳讶讹讼诀弛阱驮驯纫玖玛韧抠扼汞扳抡坎坞抑拟抒芙芜苇芥芯芭杖杉巫杈甫匣轩卤肖吱吠呕呐吟呛吻吭邑囤吮"
    "岖牡佑佃伺囱肛肘甸狈鸠彤灸刨庇吝庐闰兑灼沐沛汰沥沦汹沧沪忱诅诈罕屁坠妓姊妒纬玫卦坷坯拓坪坤拄拧拂拙"
    "拇拗茉昔苛苫苟苞茁苔枉枢枚枫杭郁矾奈奄殴歧卓昙哎咕呵咙呻咒咆咖帕账贬贮氛秉岳侠侥侣侈卑刽刹肴觅忿"
    "瓮肮肪狞庞疟疙疚卒氓炬沽沮泣泞泌沼怔怯宠宛衩祈诡帚屉弧弥陋陌函姆虱叁绅驹绊绎契贰玷玲珊拭拷拱挟垢垛"
    "拯荆茸茬荚茵茴荞荠荤荧荔栈柑栅柠枷勃柬砂泵砚鸥轴韭虐昧盹咧昵昭盅勋哆咪哟幽钙钝钠钦钧钮毡氢秕俏俄俐"
    "侯徊衍胚胧胎狰饵峦奕咨飒闺闽籽娄烁炫洼柒涎洛恃恍恬恤宦诫诬祠诲屏屎逊陨姚娜蚤骇耘耙秦匿埂捂捍袁捌"
    "挫挚捣捅埃耿聂荸莽莱莉莹莺梆栖桦栓桅桩贾酌砸砰砾殉逞哮唠哺剔蚌蚜畔蚣蚪蚓哩圃鸯唁哼唆峭唧峻赂赃钾"
    "铆氨秫笆俺赁倔殷耸舀豺豹颁胯胰脐脓逛卿鸵鸳馁凌凄衷郭斋疹紊瓷羔烙浦涡涣涤涧涕涩悍悯窍诺诽袒谆祟恕"
    "娩骏琐麸琉琅措捺捶赦埠捻掐掂掖掷掸掺勘聊娶菱菲萎菩萤乾萧萨菇彬梗梧梭曹酝酗厢硅硕奢盔匾颅彪眶晤曼"
    "晦冕啡畦趾啃蛆蚯蛉蛀唬啰唾啤啥啸崎逻崔崩婴赊铐铛铝铡铣铭矫秸秽笙笤偎傀躯兜衅徘徙舶舷舵敛翎脯逸凰"
    "猖祭烹庶庵痊阎阐眷焊焕鸿涯淑淌淮淆渊淫淳淤淀涮涵惦悴惋寂窒谍谐裆袱祷谒谓谚尉堕隅婉颇绰绷综绽缀巢"
    "琳琢琼揍堰揩揽揖彭揣搀搓壹搔葫募蒋蒂韩棱椰焚椎棺榔椭粟棘酣酥硝硫颊雳翘凿棠晰鼎喳遏晾畴跋跛蛔蜒蛤"
    "鹃喻啼喧嵌赋赎赐锉锌甥掰氮氯黍筏牍粤逾腌腋腕猩猬惫敦痘痢痪竣翔奠遂焙滞湘渤渺溃溅湃愕惶寓窖窘雇谤犀"
    "隘媒媚婿缅缆缔缕骚瑟鹉瑰搪聘斟靴靶蓖蒿蒲蓉楔椿楷榄楞楣酪碘硼碉辐辑频睹睦瞄嗜嗦暇畸跷跺蜈蜗蜕蛹嗅"
    "嗡嗤署蜀幌锚锥锨锭锰稚颓筷魁衙腻腮腺鹏肄猿颖煞雏馍馏禀痹廓痴靖誊漓溢溯溶滓溺寞窥窟寝褂裸谬媳嫉缚"
    "缤剿赘熬赫蔫摹蔓蔗蔼熙蔚兢榛榕酵碟碴碱碳辕辖雌墅嘁踊蝉嘀幔镀舔熏箍箕箫舆僧孵瘩瘟彰粹漱漩漾慷寡寥"
    "谭褐褪隧嫡缨撵撩撮撬擒墩撰鞍蕊蕴樊樟橄敷豌醇磕磅碾憋嘶嘲嘹蝠蝎蝌蝗蝙嘿幢镊镐稽篓膘鲤鲫褒瘪瘤瘫凛"
    "澎潭潦澳潘澈澜澄憔懊憎翩褥谴鹤憨履嬉豫缭撼擂擅蕾薛薇擎翰噩橱橙瓢蟥霍霎辙冀踱蹂蟆螃螟噪鹦黔穆篡"
    "篷篙篱儒膳鲸瘾瘸糙燎濒憾懈窿缰壕藐檬檐檩檀礁磷瞭瞬瞳瞪曙蹋蟋蟀嚎赡镣魏簇儡徽爵朦臊鳄糜癌懦豁臀"
    "藕藤瞻嚣鳍癞瀑襟璧戳攒孽蘑藻鳖蹭蹬簸簿蟹靡癣羹鬓攘蠕巍鳞糯譬霹躏髓蘸镶瓤矗"
)
_COMMON_SET: frozenset[str] = frozenset(_COMMON_2500)
_LESS_COMMON_SET: frozenset[str] = frozenset(_LESS_COMMON_1000)


def list_font_files(font_dir: str | Path) -> list[Path]:
    font_dir = Path(font_dir)
    return sorted(p for p in font_dir.glob("*") if p.suffix.lower() in FONT_SUFFIXES)


@lru_cache(maxsize=None)
def _font_codepoints(font_path: str) -> frozenset[int]:
    """Best cmap of a font, restricted to CHARSET_RANGES."""
    try:
        from fontTools.ttLib import TTFont
    except ImportError as exc:  # pragma: no cover - guarded by requirements
        raise RuntimeError("fontTools is required for synthesis: pip install fonttools") from exc

    font = TTFont(font_path, fontNumber=0, lazy=True)
    try:
        cmap = font.getBestCmap() or {}
    finally:
        font.close()
    allowed = set()
    for lo, hi in CHARSET_RANGES:
        allowed.update(cp for cp in cmap if lo <= cp <= hi)
    return frozenset(allowed)


@lru_cache(maxsize=256)
def _load_font(font_path: str, size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(font_path, size)


class FontPool:
    """Scans a directory of fonts and answers glyph-coverage queries."""

    def __init__(self, font_dir: str | Path) -> None:
        self.font_dir = Path(font_dir)
        self.paths = list_font_files(self.font_dir)
        if not self.paths:
            raise RuntimeError(
                f"No fonts found in {self.font_dir}. Run: python3 scripts/download_fonts.py"
            )
        self.coverage: dict[Path, frozenset[int]] = {
            path: _font_codepoints(str(path)) for path in self.paths
        }

    def __len__(self) -> int:
        return len(self.paths)

    def supports(self, path: Path, text: str) -> bool:
        cover = self.coverage[path]
        return all(ord(ch) in cover for ch in text)

    def candidates(self, text: str) -> list[Path]:
        return [path for path in self.paths if self.supports(path, text)]

    def pick(self, rng: random.Random, text: str) -> Path | None:
        candidates = self.candidates(text)
        return rng.choice(candidates) if candidates else None

    def build_charset(self, min_fonts: int = 6) -> str:
        """Characters covered by at least ``min_fonts`` fonts (clamped to pool size).

        Noto/霞鹜文楷这类全集字体覆盖全部 CJK 区间,手写体只覆盖常用字;
        提高 min_fonts 即可把字符集收紧到常用字附近。
        """
        threshold = max(1, min(int(min_fonts), len(self.paths)))
        counts: dict[int, int] = {}
        for cover in self.coverage.values():
            for cp in cover:
                counts[cp] = counts.get(cp, 0) + 1
        return "".join(sorted(chr(cp) for cp, n in counts.items() if n >= threshold))


class TextSampler:
    """Samples text lines from an optional corpus plus random generation modes."""

    def __init__(
        self,
        charset: str,
        max_text_length: int = 24,
        corpus_path: str | Path | None = None,
        cfg: dict[str, Any] | None = None,
    ) -> None:
        self.cfg = {**DEFAULT_TEXT_CFG, **(cfg or {})}
        self.max_text_length = int(max_text_length)
        chars = set(charset)
        self.charset = "".join(sorted(chars))
        self.cjk = "".join(ch for ch in self.charset if ord(ch) >= 0x4E00)
        self.ascii_alnum = "".join(ch for ch in self.charset if ch.isascii() and ch.isalnum())
        self.digits = "".join(ch for ch in self.charset if ch.isdigit())
        self.punct = "".join(
            ch for ch in self.charset if not ch.isalnum() and not 0x4E00 <= ord(ch) <= 0x9FFF
        )
        self.corpus: list[str] = []
        if corpus_path:
            self.corpus = self._load_corpus(Path(corpus_path), chars)
        self._cjk_common = [ch for ch in self.cjk if ch in _COMMON_SET]
        self._cjk_less_common = [ch for ch in self.cjk if ch in _LESS_COMMON_SET]
        self._cjk_rare = [ch for ch in self.cjk if ch not in _COMMON_SET and ch not in _LESS_COMMON_SET]

    def _sample_cjk(self, rng: random.Random) -> str:
        """Sample one CJK char with tiered frequency weighting.

        ``common_ratio`` (default 0.80): 从 2500 常用字中采样的概率.
        ``less_common_ratio`` (default 0.10): 从 1000 次常用字中采样的概率.
        剩余概率: 从全 CJK 字符集均匀采样(稀有字仍可出现,但比例可控).
        """
        r = rng.random()
        p_common = float(self.cfg.get("common_ratio", 0.80))
        p_less = float(self.cfg.get("less_common_ratio", 0.10))
        if r < p_common and self._cjk_common:
            return rng.choice(self._cjk_common)
        if r < p_common + p_less and self._cjk_less_common:
            return rng.choice(self._cjk_less_common)
        return rng.choice(list(self.cjk)) if self.cjk else "中"

    def _load_corpus(self, path: Path, allowed: set[str]) -> list[str]:
        lines: list[str] = []
        with path.open("r", encoding="utf-8") as f:
            for raw in f:
                text = "".join(ch for ch in raw.strip() if ch in allowed)
                if len(text) >= 1:
                    lines.append(text)
        return lines

    def _length(self, rng: random.Random, lo: int = 1) -> int:
        mean = float(self.cfg.get("mean_length", 6.0))
        length = lo + int(rng.expovariate(1.0 / max(mean - lo, 1.0)))
        return max(lo, min(length, self.max_text_length))

    def _random_text(self, rng: random.Random) -> str:
        weights = self.cfg.get("mode_weights", DEFAULT_TEXT_CFG["mode_weights"])
        modes, probs = zip(*[(k, float(v)) for k, v in weights.items()])
        mode = rng.choices(modes, weights=probs, k=1)[0]
        if mode == "chinese" and self.cjk:
            body = "".join(self._sample_cjk(rng) for _ in range(self._length(rng)))
        elif mode == "alnum" and self.ascii_alnum:
            body = "".join(rng.choice(self.ascii_alnum) for _ in range(self._length(rng)))
        elif mode == "digits" and self.digits:
            body = "".join(rng.choice(self.digits) for _ in range(self._length(rng, lo=2)))
        else:
            length = self._length(rng, lo=2)
            body_chars: list[str] = []
            for _ in range(length):
                if self.cjk and (not self.ascii_alnum or rng.random() < 0.6):
                    body_chars.append(self._sample_cjk(rng))
                elif self.ascii_alnum:
                    body_chars.append(rng.choice(self.ascii_alnum))
            body = "".join(body_chars) if body_chars else ""
        if self.punct and rng.random() < 0.15 and len(body) < self.max_text_length:
            body += rng.choice(self.punct)
        return body or rng.choice(self.charset)

    def sample(self, rng: random.Random) -> str:
        if self.corpus and rng.random() < float(self.cfg.get("corpus_prob", 0.7)):
            line = rng.choice(self.corpus)
            if len(line) > self.max_text_length:
                start = rng.randint(0, len(line) - self.max_text_length)
                length = rng.randint(2, self.max_text_length)
                line = line[start : start + length]
            if line:
                return line
        return self._random_text(rng)


def _random_color(rng: random.Random) -> tuple[int, int, int]:
    return (rng.randint(0, 255), rng.randint(0, 255), rng.randint(0, 255))


def _luminance(color: Iterable[int]) -> float:
    r, g, b = list(color)[:3]
    return 0.299 * r + 0.587 * g + 0.114 * b


def _contrasting_color(rng: random.Random, reference: float, min_delta: float) -> tuple[int, int, int]:
    for _ in range(24):
        color = _random_color(rng)
        if abs(_luminance(color) - reference) >= min_delta:
            return color
    return (12, 12, 12) if reference > 127 else (243, 243, 243)


class SynthTextRenderer:
    """Renders one text line into an HR image."""

    def __init__(
        self,
        font_dir: str | Path,
        hr_size: Iterable[int] | None = (128, 512),
        cfg: dict[str, Any] | None = None,
        bg_image_dir: str | Path | None = None,
    ) -> None:
        self.fonts = FontPool(font_dir)
        self.hr_size = tuple(int(v) for v in hr_size) if hr_size is not None else None
        self.cfg = {**DEFAULT_RENDER_CFG, **(cfg or {})}
        self.bg_images: list[Path] = []
        if bg_image_dir:
            bg_dir = Path(bg_image_dir)
            self.bg_images = sorted(
                p
                for p in bg_dir.glob("**/*")
                if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
            )

    # --- backgrounds --------------------------------------------------------
    # 背景先采样为"规格"(颜色参数),再物化为任意尺寸图像,这样文字颜色
    # 可以在绘制前根据真实背景亮度选取,保证对比度约束成立。
    def _background_spec(self, rng: random.Random) -> dict[str, Any]:
        weights = dict(self.cfg["background_weights"])
        if not self.bg_images:
            weights.pop("image", None)
        kinds, probs = zip(*[(k, float(v)) for k, v in weights.items()])
        kind = rng.choices(kinds, weights=probs, k=1)[0]
        if kind == "image":
            crop = self._photo_crop(rng)
            if crop is None:
                kind = "solid"
            else:
                return {"kind": "image", "crop": crop}
        spec: dict[str, Any] = {"kind": kind, "c0": _random_color(rng)}
        if kind == "gradient":
            spec["c1"] = _random_color(rng)
            spec["horizontal"] = rng.random() < 0.7
        elif kind == "noise":
            spec["sigma"] = rng.uniform(float(self.cfg["noise_sigma_min"]), float(self.cfg["noise_sigma_max"]))
            spec["noise_seed"] = rng.getrandbits(32)
        return spec

    def _spec_luminance(self, spec: dict[str, Any]) -> float:
        if spec["kind"] == "image":
            thumb = spec["crop"].resize((16, 8), Image.BILINEAR)
            return _luminance(np.asarray(thumb, dtype=np.float32).reshape(-1, 3).mean(axis=0))
        if spec["kind"] == "gradient":
            return (_luminance(spec["c0"]) + _luminance(spec["c1"])) / 2.0
        return _luminance(spec["c0"])

    def _materialize_background(self, spec: dict[str, Any], w: int, h: int) -> Image.Image:
        kind = spec["kind"]
        if kind == "image":
            return spec["crop"].resize((w, h), Image.BICUBIC)
        if kind == "gradient":
            c0 = np.array(spec["c0"], dtype=np.float32)
            c1 = np.array(spec["c1"], dtype=np.float32)
            t = np.linspace(0.0, 1.0, w if spec["horizontal"] else h, dtype=np.float32)
            ramp = c0[None, :] + t[:, None] * (c1 - c0)[None, :]
            if spec["horizontal"]:
                arr = np.broadcast_to(ramp[None, :, :], (h, w, 3))
            else:
                arr = np.broadcast_to(ramp[:, None, :], (h, w, 3))
            return Image.fromarray(np.ascontiguousarray(arr.astype(np.uint8)))
        image = Image.new("RGB", (w, h), spec["c0"])
        if kind == "noise":
            noise = np.random.default_rng(spec["noise_seed"]).normal(0.0, spec["sigma"], (h, w, 3))
            arr = np.clip(np.asarray(image, dtype=np.float32) + noise, 0, 255)
            image = Image.fromarray(arr.astype(np.uint8))
        return image

    def _photo_crop(self, rng: random.Random) -> Image.Image | None:
        path = rng.choice(self.bg_images)
        try:
            photo = Image.open(path).convert("RGB")
        except Exception:
            return None
        crop_w = rng.randint(max(1, photo.width // 4), photo.width)
        crop_h = rng.randint(max(1, photo.height // 4), photo.height)
        x0 = rng.randint(0, photo.width - crop_w)
        y0 = rng.randint(0, photo.height - crop_h)
        return photo.crop((x0, y0, x0 + crop_w, y0 + crop_h))

    # --- text layer ---------------------------------------------------------
    def _text_layer(
        self, rng: random.Random, text: str, font: ImageFont.FreeTypeFont, fill: tuple[int, int, int]
    ) -> Image.Image:
        cfg = self.cfg
        size = int(font.size)
        ascent, descent = font.getmetrics()
        stroke_width = 0
        stroke_fill = None
        if rng.random() < float(cfg["stroke_prob"]):
            stroke_width = rng.randint(1, int(cfg["stroke_width_max"]))
            stroke_fill = _contrasting_color(rng, _luminance(fill), 50.0)
        spacing = rng.uniform(float(cfg["spacing_ratio_min"]), float(cfg["spacing_ratio_max"])) * size
        jitter = float(cfg["baseline_jitter_ratio"]) * size
        margin = math.ceil(size * rng.uniform(float(cfg["margin_ratio_min"]), float(cfg["margin_ratio_max"])))
        pad = margin + stroke_width + math.ceil(jitter) + math.ceil(size * 0.08)

        advances = [max(1.0, font.getlength(ch)) for ch in text]
        total_w = sum(advances) + spacing * max(0, len(text) - 1)
        layer_w = math.ceil(total_w) + 2 * pad
        layer_h = ascent + descent + 2 * pad
        layer = Image.new("RGBA", (layer_w, layer_h), (0, 0, 0, 0))
        draw = ImageDraw.Draw(layer)

        shadow = rng.random() < float(cfg["shadow_prob"])
        shadow_offset = (rng.randint(1, max(1, size // 24)), rng.randint(1, max(1, size // 24)))
        x = float(pad)
        baseline = pad + ascent
        for ch, advance in zip(text, advances):
            y = baseline + rng.uniform(-jitter, jitter)
            if shadow:
                draw.text(
                    (x + shadow_offset[0], y + shadow_offset[1]),
                    ch,
                    font=font,
                    fill=(0, 0, 0, 140),
                    anchor="ls",
                    stroke_width=stroke_width,
                    stroke_fill=(0, 0, 0, 140) if stroke_fill else None,
                )
            draw.text(
                (x, y),
                ch,
                font=font,
                fill=(*fill, 255),
                anchor="ls",
                stroke_width=stroke_width,
                stroke_fill=(*stroke_fill, 255) if stroke_fill else None,
            )
            x += advance + spacing
        return layer

    def _warp(self, rng: random.Random, layer: Image.Image) -> Image.Image:
        cfg = self.cfg
        angle = rng.uniform(-float(cfg["rotation_max_deg"]), float(cfg["rotation_max_deg"]))
        layer = layer.rotate(angle, resample=Image.BICUBIC, expand=True)
        if rng.random() < float(cfg["perspective_prob"]):
            w, h = layer.size
            jx, jy = w * float(cfg["perspective_jitter_ratio"]), h * float(cfg["perspective_jitter_ratio"])
            quad = (
                rng.uniform(0, jx), rng.uniform(0, jy),
                rng.uniform(0, jx), h - rng.uniform(0, jy),
                w - rng.uniform(0, jx), h - rng.uniform(0, jy),
                w - rng.uniform(0, jx), rng.uniform(0, jy),
            )
            layer = layer.transform((w, h), Image.QUAD, quad, resample=Image.BICUBIC)
        return layer

    # --- main entry ---------------------------------------------------------
    def render(self, text: str, rng: random.Random) -> Image.Image | None:
        """Render one HR text-line image; None if no font covers ``text``."""
        font_path = self.fonts.pick(rng, text)
        if font_path is None:
            return None
        cfg = self.cfg
        size = rng.randint(int(cfg["font_size_min"]), int(cfg["font_size_max"]))
        font = _load_font(str(font_path), size)

        spec = self._background_spec(rng)
        fill = _contrasting_color(rng, self._spec_luminance(spec), float(cfg["contrast_min"]))

        layer = self._text_layer(rng, text, font, fill)
        layer = self._warp(rng, layer)

        canvas_h = layer.height
        min_w = math.ceil(canvas_h * rng.uniform(float(cfg["min_aspect_ratio"]), float(cfg["max_aspect_ratio"])))
        canvas_w = max(layer.width, min_w)
        background = self._materialize_background(spec, canvas_w, canvas_h)
        x = rng.randint(0, canvas_w - layer.width)
        background.paste(layer, (x, 0), layer)

        if rng.random() < float(cfg["final_blur_prob"]):
            background = background.filter(
                ImageFilter.GaussianBlur(radius=rng.uniform(0.1, float(cfg["final_blur_max"])))
            )
        if self.hr_size is not None:
            background = background.resize((self.hr_size[1], self.hr_size[0]), Image.BICUBIC)
        return background


class SynthRenderDataset(Dataset):
    """Online font-rendered pretraining dataset; LR 由在线退化生成."""

    def __init__(
        self,
        length: int,
        hr_size: Iterable[int],
        scale: int,
        font_dir: str | Path,
        max_text_length: int = 24,
        charset_min_fonts: int = 6,
        corpus_path: str | Path | None = None,
        text_cfg: dict[str, Any] | None = None,
        render_cfg: dict[str, Any] | None = None,
        bg_image_dir: str | Path | None = None,
        degradation_cfg: dict[str, Any] | None = None,
        seed: int = 0,
    ) -> None:
        from .data import degrade_tensor, pil_to_tensor  # 延迟导入避免循环依赖

        self._degrade = degrade_tensor
        self._to_tensor = pil_to_tensor
        self.length = int(length)
        self.hr_size = tuple(int(v) for v in hr_size)
        self.scale = int(scale)
        self.max_text_length = int(max_text_length)
        self.degradation_cfg = degradation_cfg or {}
        self.seed = int(seed)
        self.renderer = SynthTextRenderer(
            font_dir, hr_size=self.hr_size, cfg=render_cfg, bg_image_dir=bg_image_dir
        )
        charset = self.renderer.fonts.build_charset(charset_min_fonts)
        self.sampler = TextSampler(
            charset, max_text_length=self.max_text_length, corpus_path=corpus_path, cfg=text_cfg
        )

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx: int) -> dict[str, Any]:
        rng = random.Random((self.seed << 32) ^ idx)
        image: Image.Image | None = None
        text = ""
        for _ in range(8):
            text = self.sampler.sample(rng)
            image = self.renderer.render(text, rng)
            if image is not None:
                break
        if image is None:  # 字符集来自字体覆盖,正常不会发生
            raise RuntimeError(f"No font covers sampled text: {text!r}")
        hr = self._to_tensor(image)
        lr = self._degrade(hr, self.scale, self.degradation_cfg)
        return {"hr": hr, "lr": lr, "text": text, "id": str(idx)}
