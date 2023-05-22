log_print("init")

-- 定义播放类
chordGen = {
    env = nil,
    status = nil,
    seg = false,
    endMode = false,
    chord_name_major = {'I', '0', 'II', '0', 'III', 'IV', '0', 'V', '0', 'VI', '0', 'VII'},
    chord_name_minor = {'III', '0', 'IV', '0', 'V', 'VI', '0', 'VII', '0', 'I', '0', 'II'}
}

function chordGen:new(o, env)
    o = o or {}
    setmetatable(o, self)
    self.__index = self
    self.env = env
    return o
end

function chordGen:isRefrain()
    return self.status.averDelta > 1
end

function chordGen:isEndChord()
    local c0 = self.chord_name_major[getChordHistory(self.env, 0) % 12]
    local c1 = self.chord_name_major[getChordHistory(self.env, 1) % 12]
    local c2 = self.chord_name_major[getChordHistory(self.env, 2) % 12]
    local c3 = self.chord_name_major[getChordHistory(self.env, 3) % 12]
    -- 第一类：正格终止(Perfect Cadence) 即V->I的和弦进行
    if c0 == 'I' and c1 == 'V' and (c2 == 'II' or c2 == 'IV' or c2 == 'VI') then
        return true
    end
    -- 第二类：变格终止(Plagal Cadence) 即IV->I的和弦进行
    if (c1 == 'IV' or c1 == 'II' or c1 == 'VI') and c0 == 'I' then
        return true
    end
    -- 第三类：阻碍终止(Interrputed Cadencce) V7->I 被 V7->VI代替
    if (c3 == 'V' and c2 == 'I') and (c1 == 'V' and c0 == 'VI') then
        return true
    end
    -- 第四类：半终止(Semi Cadence) 任意和弦到V或VII（配合斜率）
    if c0 == 'V' or c0 == 'VII' then
        return true
    end
    return false
end

function chordGen:waitTime(id)
    while true do
        self.status = sleepSec(self.env)
        -- log_print("id:"..status.fragId)
        if (self.status.fragId % (self.status.bps * 16)) == id then
            --log_print("wait:" .. id .. " status:" .. cjson.encode(self.status))
            local endMode = self:isEndChord()
            self.endMode = endMode or self.endMode
            return self.status
        end
    end
end

function chordGen:playIndex(id,level,channel)
    local vel = math.floor(16*(math.random()+level-1))
    if level<=0 or vel<=0 then
        vel = 0
    elseif vel>=127 then
        vel = 127
    end
    playIndex(self.env, id, vel, channel)
end

function chordGen:play_m_1()
    -- 主歌钢琴1：最舒缓的弹奏模式，和弦中低音部分弹二分音符，高音部分弹四分音符。具体格式如下：
    setIns(self.env, 1, 0) -- 设置一号通道为钢琴
    self:playIndex(1, 5, 1) -- [1,0,2,Piano,p5]
    self:playIndex(2, 5, 1) -- [2,0,1,Piano,p5]
    self:playIndex(3, 5, 1) -- [3,0,1,Piano,p5]
    self:playIndex(4, 5, 1) -- [4,0,1,Piano,p5]
    self:playIndex(5, 5, 1) -- [5,0,1,Piano,p5]
    self:waitTime(16)
    self:playIndex(2, 4, 1) -- [2,1,1,Piano,p4]
    self:playIndex(3, 4, 1) -- [3,1,1,Piano,p4]
    self:playIndex(4, 4, 1) -- [4,1,1,Piano,p4]
    self:playIndex(5, 4, 1) -- [5,1,1,Piano,p4]
    self:waitTime(32)
    self:playIndex(1, 5, 1) -- [1,2,2,Piano,p5]
    self:playIndex(2, 4, 1) -- [2,2,1,Piano,p4]
    self:playIndex(3, 4, 1) -- [3,2,1,Piano,p4]
    self:playIndex(4, 4, 1) -- [4,2,1,Piano,p4]
    self:playIndex(5, 4, 1) -- [5,2,1,Piano,p4]
    self:waitTime(48)
    self:playIndex(2, 5, 1) -- [2,3,1,Piano,p5]
    self:playIndex(3, 5, 1) -- [3,3,1,Piano,p5]
    self:playIndex(4, 5, 1) -- [4,3,1,Piano,p5]
    self:playIndex(5, 5, 1) -- [5,3,1,Piano,p5]
end

function chordGen:play_m_2()
    -- 主歌钢琴2：模仿流行音乐中电贝司的弹法。
    -- 在主歌钢琴1的基础上，每小节最后半拍加一个和弦最低音的八分音符。具体格式如下：
    setIns(self.env, 1, 0) -- 设置一号通道为钢琴
    self:playIndex(1, 5, 1) -- [1,0,2,Piano,p5]
    self:playIndex(2, 5, 1) -- [2,0,1,Piano,p5]
    self:playIndex(3, 5, 1) -- [3,0,1,Piano,p5]
    self:playIndex(4, 5, 1) -- [4,0,1,Piano,p5]
    self:playIndex(5, 5, 1) -- [5,0,1,Piano,p5]
    self:waitTime(16)
    self:playIndex(2, 4, 1) -- [2,1,1,Piano,p4]
    self:playIndex(3, 4, 1) -- [3,1,1,Piano,p4]
    self:playIndex(4, 4, 1) -- [4,1,1,Piano,p4]
    self:playIndex(5, 4, 1) -- [5,1,1,Piano,p4]
    self:waitTime(24)
    self:playIndex(1, 4, 1) -- [1,1.5,0.5,Piano,p4]
    self:waitTime(32)
    self:playIndex(1, 5, 1) -- [1,2,2,Piano,p5]
    self:playIndex(2, 4, 1) -- [2,2,1,Piano,p4]
    self:playIndex(3, 4, 1) -- [3,2,1,Piano,p4]
    self:playIndex(4, 4, 1) -- [4,2,1,Piano,p4]
    self:playIndex(5, 4, 1) -- [5,2,1,Piano,p4]
    self:waitTime(48)
    self:playIndex(2, 5, 1) -- [2,3,1,Piano,p5]
    self:playIndex(3, 5, 1) -- [3,3,1,Piano,p5]
    self:playIndex(4, 5, 1) -- [4,3,1,Piano,p5]
    self:playIndex(5, 5, 1) -- [5,3,1,Piano,p5]
    self:waitTime(56)
    self:playIndex(1, 4, 1) -- [1,3.5,0.5,Piano,p4]
end

function chordGen:play_m_3()
    -- 主歌钢琴3：主歌太长时，将小节中第一、二、四拍的第二个音改为后半拍开始。具体格式如下：
    setIns(self.env, 1, 0) -- 设置一号通道为钢琴
    self:playIndex(1, 5, 1) -- [1,0,2,Piano,p5]
    self:playIndex(3, 5, 1) -- [3,0,1,Piano,p5]
    self:playIndex(4, 5, 1) -- [4,0,1,Piano,p5]
    self:playIndex(5, 5, 1) -- [5,0,1,Piano,p5]
    self:waitTime(8)
    self:playIndex(2, 5, 1) -- [2,0.5,1,Piano,p5]
    self:waitTime(16)
    self:playIndex(3, 5, 1) -- [3,1,1,Piano,p5]
    self:playIndex(4, 5, 1) -- [4,1,1,Piano,p5]
    self:playIndex(5, 5, 1) -- [5,1,1,Piano,p5]
    self:waitTime(24)
    self:playIndex(2, 4, 1) -- [2,1.5,0.5,Piano,p4]
    self:waitTime(32)
    self:playIndex(1, 5, 1) -- [1,2,2,Piano,p5]
    self:playIndex(2, 4, 1) -- [2,2,1.5,Piano,p4]
    self:playIndex(3, 4, 1) -- [3,2,1,Piano,p4]
    self:playIndex(4, 4, 1) -- [4,2,1,Piano,p4]
    self:playIndex(5, 4, 1) -- [5,2,1,Piano,p4]
    self:waitTime(48)
    self:playIndex(3, 5, 1) -- [3,3,1,Piano,p5]
    self:playIndex(4, 5, 1) -- [4,3,1,Piano,p5]
    self:playIndex(5, 5, 1) -- [5,3,1,Piano,p5]
    self:waitTime(56)
    self:playIndex(2, 4, 1) -- [2,3.5,0.5,Piano,p4]
end

function chordGen:play_m_s()
    -- 终止和弦
    -- 主歌钢琴加花：
    setIns(self.env, 1, 0) -- 设置一号通道为钢琴
    -- 主歌乐句过渡处，在主歌钢琴2的基础上，
    -- 最后一拍取代为以十六分音符弹当前和弦高八度的1232音。具体格式如下：

    self:playIndex(1, 5, 1) -- [1,0,2,Piano,p5]
    self:playIndex(2, 5, 1) -- [2,0,1,Piano,p5]
    self:playIndex(3, 5, 1) -- [3,0,1,Piano,p5]
    self:playIndex(4, 5, 1) -- [4,0,1,Piano,p5]
    self:playIndex(5, 5, 1) -- [5,0,1,Piano,p5]

    self:waitTime(16)
    self:playIndex(2, 4, 1) -- [2,1,1,Piano,p4]
    self:playIndex(3, 4, 1) -- [3,1,1,Piano,p4]
    self:playIndex(4, 4, 1) -- [4,1,1,Piano,p4]
    self:playIndex(5, 4, 1) -- [5,1,1,Piano,p4]

    self:waitTime(24)
    self:playIndex(1, 4, 1) -- [1,1.5,0.5,Piano,p4]

    self:waitTime(32)
    self:playIndex(1, 5, 1) -- [1,2,2,Piano,p5]
    self:playIndex(2, 4, 1) -- [2,2,1,Piano,p4]
    self:playIndex(3, 4, 1) -- [3,2,1,Piano,p4]
    self:playIndex(4, 4, 1) -- [4,2,1,Piano,p4]
    self:playIndex(5, 4, 1) -- [5,2,1,Piano,p4]

    self:waitTime(48)
    self:playIndex(1, 5, 1) -- [1,3,1,Piano,p5]

    self:waitTime(52)
    self:playIndex(2, 4, 1) -- [2,3.25,0.75,Piano,p4]

    self:waitTime(56)
    self:playIndex(3, 6, 1) -- [3,3.5,0.5,Piano,p6]

    self:waitTime(60)
    self:playIndex(2, 4, 1) -- [2,3.75,0.25,Piano,p4]
end

function chordGen:play_r_1_1()
    -- 副歌钢琴1：每小节的第一、二拍的第二个音变为后半拍开始；
    -- 每小节第二拍的最后四分之一拍添加一个根音；
    -- 每小节第三拍后半拍添加当前和弦第二个音；
    -- 每小节第三拍的最后四分之一拍添加当前和弦所有除了根音之外的16分音符；
    -- 每小节最后一拍第一个16分空掉，后三个依次弹当前和弦高八度的321音；
    -- 第四小节最后一拍的三个音的顺序改成123音（Cmaj7就是CEG）。
    -- 各小节具体格式如下：
    -- 第一小节：
    setIns(self.env, 1, 0) -- 设置一号通道为钢琴
    setIns(self.env, 2, 24) -- 设置二号通道为吉他

    self:playIndex(1, 6, 1) -- [1,0,1.75,Piano,p6]
    self:playIndex(3, 6, 1) -- [3,0,1,Piano,p6]
    self:playIndex(4, 6, 1) -- [4,0,1,Piano,p6]
    self:playIndex(5, 6, 1) -- [5,0,1,Piano,p6]
    self:waitTime(8)
    self:playIndex(2, 4, 1) -- [2,0.5,1,Piano,p4]
    self:playIndex(1, 5, 2) -- [1,0.5,1.25,Guitar,g5]
    self:waitTime(12)
    self:playIndex(2, 6, 2) -- [2,0.75,1.25,Guitar,g6]
    self:waitTime(16)
    self:playIndex(3, 5, 1) -- [3,1,1,Piano,p5]
    self:playIndex(4, 5, 1) -- [4,1,1,Piano,p5]
    self:playIndex(5, 5, 1) -- [5,1,1,Piano,p5]
    self:waitTime(20)
    self:playIndex(3, 7, 2) -- [3,1.25,0.75,Guitar,g7]
    self:waitTime(24)
    self:playIndex(2, 5, 1) -- [2,1.5,0.5,Piano,p5]
    self:waitTime(28)
    self:playIndex(1, 5, 1) -- [1,1.75,0.25,Piano,p5]
    self:playIndex(1, 5, 2) -- [1,1.75,0.25,Guitar,g5]
    self:waitTime(32)
    self:playIndex(1, 6, 1) -- [1,2,2,Piano,p6]
    self:playIndex(2, 5, 1) -- [2,2,0.5,Piano,p5]
    self:playIndex(3, 6, 1) -- [3,2,0.75,Piano,p6]
    self:playIndex(4, 6, 1) -- [4,2,0.75,Piano,p6]
    self:playIndex(5, 6, 1) -- [5,2,0.75,Piano,p6]
    self:playIndex(1, 6, 2) -- [1,2,1.75,Guitar,g6]
    self:playIndex(3, 6, 2) -- [3,2,0.75,Guitar,g6]
    self:waitTime(40)
    self:playIndex(2, 4, 1) -- [2,2.5,0.25,Piano,p4]
    self:playIndex(2, 4, 1) -- [2,2.5,0.75,Guitar,g4]
    self:waitTime(44)
    self:playIndex(4, 6, 1) -- [4,2.75,1.25,Piano,p6]
    self:playIndex(5, 6, 1) -- [5,2.75,1.25,Piano,p6]
    self:playIndex(2, 6, 1) -- [2,2.75,1.25,Piano,p6]
    self:playIndex(3, 6, 1) -- [3,2.75,1.25,Piano,p6]
    self:playIndex(3, 6, 2) -- [3,2.75,1.25,Guitar,g6]
    self:waitTime(52)
    self:playIndex(3, 6, 1) -- [3,3.25,0.75,Piano,p6]
    self:playIndex(2, 5, 2) -- [2,3.25,0.75,Guitar,g5]
    self:waitTime(56)
    self:playIndex(2, 5, 1) -- [2,3.5,0.5,Piano,p5]
    self:waitTime(60)
    self:playIndex(1, 5, 1) -- [1,3.75,0.25,Piano,p5]
    self:playIndex(1, 5, 2) -- [1,3.75,0.25,Guitar,g5]
end

function chordGen:play_r_1_2()
    -- 第二小节：
    setIns(self.env, 1, 0) -- 设置一号通道为钢琴
    setIns(self.env, 2, 24) -- 设置二号通道为吉他
    self:playIndex(1, 6, 1) -- [1,0,2,Piano,p6]
    self:playIndex(3, 6, 1) -- [3,0,1,Piano,p6]
    self:playIndex(4, 6, 1) -- [4,0,1,Piano,p6]
    self:playIndex(5, 6, 1) -- [5,0,1,Piano,p6]
    self:playIndex(1, 5, 2) -- [1,0,1.75,Guitar,g5]
    self:waitTime(8)
    self:playIndex(2, 4, 1) -- [2,0.5,1,Piano,p4]
    self:waitTime(12)
    self:playIndex(2, 6, 2) -- [2,0.75,1.25,Guitar,g6]
    self:waitTime(16)
    self:playIndex(3, 5, 1) -- [3,1,1,Piano,p5]
    self:playIndex(4, 5, 1) -- [4,1,1,Piano,p5]
    self:playIndex(5, 5, 1) -- [5,1,1,Piano,p5]
    self:waitTime(20)
    self:playIndex(3, 7, 2) -- [3,1.25,0.75,Guitar,g7]
    self:waitTime(24)
    self:playIndex(2, 5, 1) -- [2,1.5,0.5,Piano,p5]
    self:waitTime(28)
    self:playIndex(1, 5, 1) -- [1,1.75,0.25,Piano,p5]
    self:playIndex(1, 4, 2) -- [1,1.75,0.25,Guitar,g4]
    self:waitTime(32)
    self:playIndex(1, 6, 1) -- [1,2,2,Piano,p6]
    self:playIndex(2, 5, 1) -- [2,2,0.5,Piano,p5]
    self:playIndex(3, 6, 1) -- [3,2,0.75,Piano,p6]
    self:playIndex(4, 6, 1) -- [4,2,0.75,Piano,p6]
    self:playIndex(5, 6, 1) -- [5,2,0.75,Piano,p6]
    self:playIndex(2, 5, 2) -- [2,2,0.75,Guitar,g5]
    self:waitTime(40)
    self:playIndex(2, 4, 1) -- [2,2.5,0.25,Piano,p4]
    self:playIndex(1, 5, 2) -- [1,2.5,1.5,Guitar,g5]
    self:waitTime(44)
    self:playIndex(2, 6, 1) -- [2,2.75,1.25,Piano,p6]
    self:playIndex(3, 6, 1) -- [3,2.75,1.25,Piano,p6]
    self:playIndex(4, 6, 1) -- [4,2.75,1.25,Piano,p6]
    self:playIndex(5, 6, 1) -- [5,2.75,1.25,Piano,p6]
    self:playIndex(2, 5, 2) -- [2,2.75,1.25,Guitar,g5]
    self:waitTime(52)
    self:playIndex(3, 6, 1) -- [3,3.25,0.75,Piano,p6]
    self:playIndex(3, 6, 2) -- [3,3.25,0.75,Guitar,g6]
    self:waitTime(56)
    self:playIndex(2, 5, 1) -- [2,3.5,0.5,Piano,p5]
    self:waitTime(60)
    self:playIndex(1, 5, 1) -- [1,3.75,0.25,Piano,p5]
end

function chordGen:play_r_1_3()
    -- 第三小节：
    setIns(self.env, 1, 0) -- 设置一号通道为钢琴
    setIns(self.env, 2, 24) -- 设置二号通道为吉他
    self:playIndex(1, 6, 1) -- [1,0,2,Piano,p6]
    self:playIndex(3, 6, 1) -- [3,0,1,Piano,p6]
    self:playIndex(4, 6, 1) -- [4,0,1,Piano,p6]
    self:playIndex(5, 6, 1) -- [5,0,1,Piano,p6]
    self:playIndex(2, 6, 2) -- [2,0,0.75,Guitar,g6]
    self:waitTime(8)
    self:playIndex(2, 4, 1) -- [2,0.5,1,Piano,p4]
    self:playIndex(1, 5, 2) -- [1,0.5,1.25,Guitar,g5]
    self:waitTime(12)
    self:playIndex(2, 6, 2) -- [2,0.75,1.25,Guitar,g6]
    self:waitTime(16)
    self:playIndex(3, 5, 1) -- [3,1,1,Piano,p5]
    self:playIndex(4, 5, 1) -- [4,1,1,Piano,p5]
    self:playIndex(5, 5, 1) -- [5,1,1,Piano,p5]
    self:waitTime(20)
    self:playIndex(3, 7, 2) -- [3,1.25,0.75,Guitar,g7]
    self:waitTime(24)
    self:playIndex(2, 5, 1) -- [2,1.5,0.5,Piano,p5]
    self:waitTime(28)
    self:playIndex(1, 5, 1) -- [1,1.75,0.75,Piano,p5]
    self:playIndex(1, 5, 2) -- [1,1.75,0.25,Guitar,g5]
    self:waitTime(32)
    self:playIndex(1, 6, 1) -- [1,2,2,Piano,p6]
    self:playIndex(3, 6, 1) -- [3,2,0.75,Piano,p6]
    self:playIndex(2, 5, 1) -- [2,2,0.5,Piano,p5]
    self:playIndex(4, 6, 1) -- [4,2,0.75,Piano,p6]
    self:playIndex(5, 6, 1) -- [5,2,0.75,Piano,p6]
    self:waitTime(36)
    self:playIndex(1, 7, 2) -- [1,2.25,1.5,Guitar,g7]
    self:playIndex(2, 7, 2) -- [2,2.25,1,Guitar,g7]
    self:waitTime(40)
    self:playIndex(2, 4, 1) -- [2,2.5,0.25,Piano,p4]
    self:waitTime(44)
    self:playIndex(2, 7, 1) -- [2,2.75,0.25,Piano,p7]
    self:playIndex(3, 7, 1) -- [3,2.75,0.25,Piano,p7]
    self:playIndex(4, 7, 1) -- [4,2.75,0.25,Piano,p7]
    self:playIndex(5, 7, 1) -- [5,2.75,0.25,Piano,p7]
    self:playIndex(3, 7, 2) -- [3,2.75,1.25,Guitar,g7]
    self:waitTime(52)
    self:playIndex(3, 7, 1) -- [3,3.25,0.75,Piano,p7]
    self:playIndex(2, 5, 2) -- [2,3.25,0.75,Guitar,g5]
    self:waitTime(56)
    self:playIndex(2, 6, 1) -- [2,3.5,0.5,Piano,p6]
    self:waitTime(60)
    self:playIndex(1, 5, 1) -- [1,3.75,0.25,Piano,p5]
    self:playIndex(1, 5, 2) -- [1,3.75,0.25,Guitar,g5]
end

function chordGen:play_r_1_4()
    -- 第四小节：
    setIns(self.env, 1, 0) -- 设置一号通道为钢琴
    setIns(self.env, 2, 24) -- 设置二号通道为吉他
    self:playIndex(1, 6, 1) -- [1,0,2,Piano,p6]
    self:playIndex(3, 6, 1) -- [3,0,1,Piano,p6]
    self:playIndex(4, 6, 1) -- [4,0,1,Piano,p6]
    self:playIndex(5, 6, 1) -- [5,0,1,Piano,p6]
    self:playIndex(1, 6, 2) -- [1,0,1.75,Guitar,g6]
    self:waitTime(8)
    self:playIndex(2, 5, 1) -- [2,0.5,1,Piano,p5]
    self:waitTime(12)
    self:playIndex(2, 6, 2) -- [2,0.75,1.25,Guitar,g6]
    self:waitTime(16)
    self:playIndex(3, 7, 1) -- [3,1,1,Piano,p7]
    self:playIndex(4, 7, 1) -- [4,1,1,Piano,p7]
    self:playIndex(5, 7, 1) -- [5,1,1,Piano,p7]
    self:waitTime(20)
    self:playIndex(3, 7, 2) -- [3,1.25,0.75,Guitar,g7]
    self:waitTime(24)
    self:playIndex(2, 6, 1) -- [2,1.5,0.5,Piano,p6]
    self:waitTime(28)
    self:playIndex(1, 6, 1) -- [1,1.75,0.25,Piano,p6]
    self:playIndex(1, 5, 2) -- [1,1.75,0.25,Guitar,g5]
    self:waitTime(32)
    self:playIndex(1, 7, 1) -- [1,2,2,Piano,p7]
    self:playIndex(2, 6, 1) -- [2,2,0.5,Piano,p6]
    self:playIndex(3, 6, 1) -- [3,2,0.75,Piano,p6]
    self:playIndex(4, 6, 1) -- [4,2,0.75,Piano,p6]
    self:playIndex(5, 6, 1) -- [5,2,0.75,Piano,p6]
    self:playIndex(1, 5, 2) -- [1,2,1.75,Guitar,g5]
    self:waitTime(40)
    self:playIndex(2, 5, 1) -- [2,2.5,0.25,Piano,p5]
    self:playIndex(2, 6, 2) -- [2,2.5,1,Guitar,g6]
    self:waitTime(44)
    self:playIndex(2, 7, 1) -- [2,2.75,1.25,Piano,p7]
    self:playIndex(3, 7, 1) -- [3,2.75,1.25,Piano,p7]
    self:playIndex(4, 7, 1) -- [4,2.75,1.25,Piano,p7]
    self:playIndex(5, 7, 1) -- [5,2.75,1.25,Piano,p7]
    self:playIndex(3, 7, 2) -- [3,2.75,1.25,Guitar,g7]
    self:waitTime(52)
    self:playIndex(1, 6, 1) -- [1,3.25,0.75,Piano,p6]
    self:playIndex(2, 6, 2) -- [2,3.25,0.75,Guitar,g6]
    self:waitTime(56)
    self:playIndex(2, 7, 1) -- [2,3.5,0.5,Piano,p7]
    self:waitTime(60)
    self:playIndex(3, 7, 1) -- [3,3.75,0.25,Piano,p7]
    self:playIndex(1, 5, 2) -- [1,3.75,0.25,Guitar,g5]
end

function chordGen:play_r_2_1()
    -- 副歌钢琴2：是副歌钢琴1的变形，情绪上更加递进一些。各小节具体格式如下：
    -- 第一小节：
    setIns(self.env, 1, 0) -- 设置一号通道为钢琴
    setIns(self.env, 2, 24) -- 设置二号通道为吉他
    self:playIndex(1, 7, 1) -- [1,0,2,Piano,p7]
    self:playIndex(3, 6, 1) -- [3,0,1,Piano,p6]
    self:playIndex(4, 6, 1) -- [4,0,1,Piano,p6]
    self:playIndex(5, 6, 1) -- [5,0,1,Piano,p6]
    self:waitTime(8)
    self:playIndex(1, 5, 2) -- [1,0.5,1.25,Guitar,g5]
    self:waitTime(12)
    self:playIndex(2, 5, 1) -- [2,0.75,0.75,Piano,p5]
    self:playIndex(2, 6, 2) -- [2,0.75,1.25,Guitar,g6]
    self:waitTime(16)
    self:playIndex(3, 6, 1) -- [3,1,0.75,Piano,p6]
    self:playIndex(4, 6, 1) -- [4,1,0.75,Piano,p6]
    self:playIndex(5, 6, 1) -- [5,1,0.75,Piano,p6]
    self:waitTime(20)
    self:playIndex(3, 7, 2) -- [3,1.25,0.75,Guitar,g7]
    self:waitTime(24)
    self:playIndex(2, 5, 1) -- [2,1.5,0.25,Piano,p5]
    self:waitTime(28)
    self:playIndex(2, 7, 1) -- [2,1.75,0.25,Piano,p7]
    self:playIndex(3, 6, 1) -- [3,1.75,0.25,Piano,p6]
    self:playIndex(4, 6, 1) -- [4,1.75,0.25,Piano,p6]
    self:playIndex(5, 6, 1) -- [5,1.75,0.25,Piano,p6]
    self:playIndex(1, 5, 2) -- [1,1.75,0.25,Guitar,g5]
    self:waitTime(32)
    self:playIndex(1, 6, 1) -- [1,2,2,Piano,p6]
    self:playIndex(1, 6, 2) -- [1,2,1.75,Guitar,g6]
    self:playIndex(3, 6, 2) -- [3,2,0.75,Guitar,g6]
    self:waitTime(36)
    self:playIndex(2, 5, 1) -- [2,2.25,0.25,Piano,p5]
    self:playIndex(3, 5, 1) -- [3,2.25,0.5,Piano,p5]
    self:playIndex(4, 5, 1) -- [4,2.25,0.5,Piano,p5]
    self:playIndex(5, 5, 1) -- [5,2.25,0.5,Piano,p5]
    self:waitTime(40)
    self:playIndex(2, 6, 1) -- [2,2.5,0.25,Piano,p6]
    self:playIndex(2, 4, 2) -- [2,2.5,0.75,Guitar,g4]
    self:waitTime(44)
    self:playIndex(2, 7, 1) -- [2,2.75,1.25,Piano,p7]
    self:playIndex(3, 7, 1) -- [3,2.75,1.25,Piano,p7]
    self:playIndex(4, 7, 1) -- [4,2.75,1.25,Piano,p7]
    self:playIndex(5, 7, 1) -- [5,2.75,1.25,Piano,p7]
    self:playIndex(3, 6, 2) -- [3,2.75,1.25,Guitar,g6]
    self:waitTime(52)
    self:playIndex(3, 7, 1) -- [3,3.25,0.75,Piano,p7]
    self:playIndex(2, 5, 2) -- [2,3.25,0.75,Guitar,g5]
    self:waitTime(56)
    self:playIndex(2, 6, 1) -- [2,3.5,0.5,Piano,p6]
    self:waitTime(60)
    self:playIndex(1, 5, 1) -- [1,3.75,0.25,Piano,p5]
    self:playIndex(1, 5, 2) -- [1,3.75,0.25,Guitar,g5]
end

function chordGen:play_r_2_2()
    -- 第二小节：
    setIns(self.env, 1, 0) -- 设置一号通道为钢琴
    setIns(self.env, 2, 24) -- 设置二号通道为吉他
    self:playIndex(1, 7, 1) -- [1,0,2,Piano,p7]
    self:playIndex(3, 6, 1) -- [3,0,1,Piano,p6]
    self:playIndex(4, 6, 1) -- [4,0,1,Piano,p6]
    self:playIndex(5, 6, 1) -- [5,0,1,Piano,p6]
    self:playIndex(1, 5, 2) -- [1,0,1.75,Guitar,g5]
    self:waitTime(8)
    self:playIndex(2, 5, 1) -- [2,0.5,1,Piano,p5]
    self:waitTime(12)
    self:playIndex(2, 6, 2) -- [2,0.75,1.25,Guitar,g6]
    self:waitTime(16)
    self:playIndex(3, 6, 1) -- [3,1,0.75,Piano,p6]
    self:playIndex(4, 6, 1) -- [4,1,0.75,Piano,p6]
    self:playIndex(5, 6, 1) -- [5,1,0.75,Piano,p6]
    self:waitTime(20)
    self:playIndex(3, 7, 2) -- [3,1.25,0.75,Guitar,g7]
    self:waitTime(24)
    self:playIndex(2, 5, 1) -- [2,1.5,0.25,Piano,p5]
    self:waitTime(28)
    self:playIndex(2, 6, 1) -- [2,1.75,0.25,Piano,p6]
    self:playIndex(3, 5, 1) -- [3,1.75,0.25,Piano,p5]
    self:playIndex(4, 5, 1) -- [4,1.75,0.25,Piano,p5]
    self:playIndex(5, 5, 1) -- [5,1.75,0.25,Piano,p5]
    self:playIndex(1, 4, 2) -- [1,1.75,0.25,Guitar,g4]
    self:waitTime(32)
    self:playIndex(1, 6, 1) -- [1,2,2,Piano,p6]
    self:playIndex(2, 5, 1) -- [2,2,0.5,Piano,p5]
    self:playIndex(3, 5, 1) -- [3,2,0.75,Piano,p5]
    self:playIndex(4, 5, 1) -- [4,2,0.75,Piano,p5]
    self:playIndex(5, 5, 1) -- [5,2,0.75,Piano,p5]
    self:playIndex(2, 5, 2) -- [2,2,0.75,Guitar,g5]
    self:waitTime(40)
    self:playIndex(2, 5, 1) -- [2,2.5,0.25,Piano,p5]
    self:playIndex(1, 5, 2) -- [1,2.5,1.5,Guitar,g5]
    self:waitTime(44)
    self:playIndex(2, 7, 1) -- [2,2.75,1.25,Piano,p7]
    self:playIndex(3, 7, 1) -- [3,2.75,1.25,Piano,p7]
    self:playIndex(4, 7, 1) -- [4,2.75,1.25,Piano,p7]
    self:playIndex(5, 7, 1) -- [5,2.75,1.25,Piano,p7]
    self:playIndex(2, 5, 2) -- [2,2.75,1.25,Guitar,g5]
    self:waitTime(52)
    self:playIndex(3, 7, 1) -- [3,3.25,0.75,Piano,p7]
    self:playIndex(3, 6, 2) -- [3,3.25,0.75,Guitar,g6]
    self:waitTime(56)
    self:playIndex(2, 6, 1) -- [2,3.5,0.5,Piano,p6]
    self:waitTime(60)
    self:playIndex(1, 5, 1) -- [1,3.75,0.25,Piano,p5]
end

function chordGen:play_r_2_3()
    -- 第三小节：
    setIns(self.env, 1, 0) -- 设置一号通道为钢琴
    setIns(self.env, 2, 24) -- 设置二号通道为吉他
    self:playIndex(1, 7, 1) -- [1,0,2,Piano,p7]
    self:playIndex(3, 6, 1) -- [3,0,1,Piano,p6]
    self:playIndex(4, 6, 1) -- [4,0,1,Piano,p6]
    self:playIndex(5, 6, 1) -- [5,0,1,Piano,p6]
    self:playIndex(2, 6, 2) -- [2,0,0.75,Guitar,g6]
    self:waitTime(8)
    self:playIndex(1, 5, 2) -- [1,0.5,1.25,Guitar,g5]
    self:waitTime(12)
    self:playIndex(2, 5, 1) -- [2,0.75,0.75,Piano,p5]
    self:playIndex(2, 6, 2) -- [2,0.75,1.25,Guitar,g6]
    self:waitTime(16)
    self:playIndex(3, 6, 1) -- [3,1,0.75,Piano,p6]
    self:playIndex(4, 6, 1) -- [4,1,0.75,Piano,p6]
    self:playIndex(5, 6, 1) -- [5,1,0.75,Piano,p6]
    self:waitTime(20)
    self:playIndex(3, 7, 2) -- [3,1.25,0.75,Guitar,g7]
    self:waitTime(24)
    self:playIndex(2, 5, 1) -- [2,1.5,0.25,Piano,p5]
    self:waitTime(28)
    self:playIndex(2, 7, 1) -- [2,1.75,0.25,Piano,p7]
    self:playIndex(3, 6, 1) -- [3,1.75,0.25,Piano,p6]
    self:playIndex(4, 6, 1) -- [4,1.75,0.25,Piano,p6]
    self:playIndex(5, 6, 1) -- [5,1.75,0.25,Piano,p6]
    self:playIndex(1, 5, 2) -- [1,1.75,0.25,Guitar,g5]
    self:waitTime(32)
    self:playIndex(1, 6, 1) -- [1,2,2,Piano,p6]
    self:waitTime(36)
    self:playIndex(2, 6, 1) -- [2,2.25,0.25,Piano,p6]
    self:playIndex(3, 6, 1) -- [3,2.25,0.25,Piano,p6]
    self:playIndex(4, 6, 1) -- [4,2.25,0.25,Piano,p6]
    self:playIndex(5, 6, 1) -- [5,2.25,0.25,Piano,p6]
    self:playIndex(1, 7, 2) -- [1,2.25,1.5,Guitar,g7]
    self:playIndex(2, 7, 2) -- [2,2.25,1,Guitar,g7]
    self:waitTime(44)
    self:playIndex(2, 7, 1) -- [2,2.75,1.25,Piano,p7]
    self:playIndex(3, 7, 1) -- [3,2.75,1.25,Piano,p7]
    self:playIndex(4, 7, 1) -- [4,2.75,1.25,Piano,p7]
    self:playIndex(5, 7, 1) -- [5,2.75,1.25,Piano,p7]
    self:playIndex(3, 7, 2) -- [3,2.75,1.25,Guitar,g7]
    self:waitTime(52)
    self:playIndex(3, 8, 1) -- [3,3.25,0.75,Piano,p8]
    self:waitTime(56)
    self:playIndex(2, 6, 1) -- [2,3.5,0.5,Piano,p6]
    self:waitTime(60)
    self:playIndex(1, 5, 1) -- [1,3.75,0.25,Piano,p5]
    self:playIndex(1, 5, 2) -- [1,3.75,0.25,Guitar,g5]
end

function chordGen:play_r_2_4()
    -- 第四小节：
    setIns(self.env, 1, 0) -- 设置一号通道为钢琴
    setIns(self.env, 2, 24) -- 设置二号通道为吉他
    self:playIndex(1, 7, 1) -- [1,0,2,Piano,p7]
    self:playIndex(3, 7, 1) -- [3,0,1,Piano,p7]
    self:playIndex(4, 7, 1) -- [4,0,1,Piano,p7]
    self:playIndex(5, 7, 1) -- [5,0,1,Piano,p7]
    self:playIndex(1, 6, 2) -- [1,0,1.75,Guitar,g6]
    self:waitTime(8)
    self:playIndex(2, 5, 1) -- [2,0.5,1,Piano,p5]
    self:waitTime(16)
    self:playIndex(3, 7, 1) -- [3,1,0.75,Piano,p7]
    self:playIndex(4, 7, 1) -- [4,1,0.75,Piano,p7]
    self:playIndex(5, 7, 1) -- [5,1,0.75,Piano,p7]
    self:playIndex(2, 6, 2) -- [2,0.75,1.25,Guitar,g6]
    self:waitTime(20)
    self:playIndex(3, 7, 2) -- [3,1.25,0.75,Guitar,g7]
    self:waitTime(24)
    self:playIndex(2, 5, 1) -- [2,1.5,0.25,Piano,p5]
    self:waitTime(28)
    self:playIndex(2, 6, 1) -- [2,1.75,0.25,Piano,p6]
    self:playIndex(3, 6, 1) -- [3,1.75,0.25,Piano,p6]
    self:playIndex(4, 6, 1) -- [4,1.75,0.25,Piano,p6]
    self:playIndex(5, 6, 1) -- [5,1.75,0.25,Piano,p6]
    self:playIndex(1, 5, 2) -- [1,1.75,0.25,Guitar,g5]
    self:waitTime(32)
    self:playIndex(1, 6, 1) -- [1,2,2,Piano,p6]
    self:playIndex(2, 6, 1) -- [2,2,0.5,Piano,p6]
    self:playIndex(3, 6, 1) -- [3,2,0.75,Piano,p6]
    self:playIndex(4, 6, 1) -- [4,2,0.75,Piano,p6]
    self:playIndex(5, 6, 1) -- [5,2,0.75,Piano,p6]
    self:playIndex(1, 5, 2) -- [1,2,1.75,Guitar,g5]
    self:waitTime(40)
    self:playIndex(2, 4, 1) -- [2,2.5,0.25,Piano,p4]
    self:playIndex(2, 6, 2) -- [2,2.5,1,Guitar,g6]
    self:waitTime(44)
    self:playIndex(2, 6, 1) -- [2,2.75,1.25,Piano,p6]
    self:playIndex(3, 6, 1) -- [3,2.75,1.25,Piano,p6]
    self:playIndex(4, 6, 1) -- [4,2.75,1.25,Piano,p6]
    self:playIndex(5, 6, 1) -- [5,2.75,1.25,Piano,p6]
    self:playIndex(3, 7, 2) -- [3,2.75,1.25,Guitar,g7]
    self:waitTime(52)
    self:playIndex(1, 5, 1) -- [1,3.25,0.75,Piano,p5]
    self:playIndex(2, 6, 2) -- [2,3.25,0.75,Guitar,g6]
    self:waitTime(56)
    self:playIndex(2, 5, 1) -- [2,3.5,0.5,Piano]
    self:waitTime(60)
    self:playIndex(3, 7, 1) -- [3,3.75,0.25,Piano,p7]
    self:playIndex(1, 5, 2) -- [1,3.75,0.25,Guitar,g5]
end

function chordGen:play_r_s()
    -- 终止和弦
    setIns(self.env, 1, 0) -- 设置一号通道为钢琴
    -- 副歌钢琴加花：一小节的长音。具体格式如下：
    self:playIndex(1, 6, 1) -- [1,0,4,Piano,p6]
    self:playIndex(2, 4, 1) -- [2,0,4,Piano,p4]
    self:playIndex(3, 5, 1) -- [3,0,4,Piano,p5]
    self:playIndex(4, 5, 1) -- [4,0,4,Piano,p5]
    self:playIndex(5, 5, 1) -- [5,0,4,Piano,p5]
end

function chordGen:play_r()
    local secId = math.floor(self.status.fragId / 64) % 8
    if self.endMode then
        self:play_r_s()
        self.endMode = false
    else
        if secId == 0 then
            self:play_r_1_1()
        elseif secId == 1 then
            self:play_r_1_2()
        elseif secId == 2 then
            self:play_r_1_3()
        elseif secId == 3 then
            self:play_r_1_4()
        elseif secId == 4 then
            self:play_r_2_1()
        elseif secId == 5 then
            self:play_r_2_2()
        elseif secId == 6 then
            self:play_r_2_3()
        elseif secId == 7 then
            self:play_r_2_4()
        end
    end
end

function chordGen:play_m()
    local secId = math.floor(self.status.fragId / 256) % 3
    if self.endMode then
        self:play_m_s()
        self.endMode = false
    else
        if secId == 0 then
            self:play_m_1()
        elseif secId == 1 then
            self:play_m_2()
        elseif secId == 2 then
            self:play_m_3()
        end
    end
end

function main(env)
    -- 初始化
    -- 启动自动停止音符功能
    setAutoStopAll(env, true)
    -- 初始化随机数
    -- 得到时间字符串
    local strTime = tostring(os.time())
    -- 得到一个反转字符串
    local strRev = string.reverse(strTime)
    -- 得到前6位
    local strRandomTime = string.sub(strRev, 1, 6)
    -- 设置时间种子
    math.randomseed(strRandomTime)

    local gen = chordGen:new(nil, env)

    gen:waitTime(0)
    local seg = gen:isRefrain()
    gen:play_m_1()
    gen:waitTime(0)
    gen:play_m_2()
    gen:waitTime(0)
    while true do
        seg = gen:isRefrain()
        if seg then
            gen:play_r()
        else
            gen:play_m()
        end
        gen:waitTime(0)
    end
end
