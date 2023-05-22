log_print("init")
--[[
Guitar：
一个音：
[1,0,4,Guitar,p4]
两个音：
[1,0,4,Guitar,p4],[2,0,4,Guitar,p4]
三个音：
[1,0,4,Guitar,p4],[2,0,4,Guitar,p4,+12],[3,0,4,Guitar,p4]
四个音：[1,0,4,Guitar,p4],[2,0,4,Guitar,p4,+12],[3,0,4,Guitar,p4,+12],[4,0,4,Guitar,p4]
五个音：[1,0,4,Guitar,p4],[2,0,4,Guitar,p4,+12],[3,0,4,Guitar,p4,+12],[4,0,4,Guitar,p4],[5,0,4,Guitar,p4]

Piano：
一个音：
[1,0,4,Piano,p4]
两个音：
[1,0,4,Piano,p4],[2,0,4,Piano,p4]
三个音：
[1,0,4,Piano,p4],[2,2,2,Piano,p3,+12],[3,0,4,Piano,p4]
四个音：
[1,0,4,Piano,p4],[2,2,2,Piano,p3,+12],[3,2,2,Piano,p3,+12],[4,0,4,Piano,p4]
五个音：
[1,0,4,Piano,p4],[2,2,2,Piano,p3,+12],[3,0,4,Piano,p3,+12],[4,0,4,Piano,p4],[5,0,4,Piano,p4]
]] --
function waitTime(env, id)
    while true do
        local status = sleepSec(env)
        -- log_print("id:"..status.fragId)
        if (status.fragId % (status.bps * 16)) == id then
            return status
        end
    end
end
function P(env, id, level, channel, delta)
    local vel = math.floor(16 * (math.random() + level - 1))
    if level <= 0 or vel <= 0 then
        vel = 0
    elseif vel >= 127 then
        vel = 127
    end
    playIndex(env, id, vel, channel, delta)
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

    while true do
        waitTime(env, 0)
        log_print("seg" .. playListSize(env))
        setIns(env, 1, 0) -- 设置一号通道为钢琴
        local pianoShift = shiftPlayList(env, 28)
        setIns(env, 2, 24) -- 设置二号通道为吉他
        local guitarShift = shiftPlayList(env, 40)

        print("pianoShift", pianoShift)

        if playListSize(env) == 1 then
            -- 一个音：
            P(env, 1, 4, 2, 0) -- [1,0,4,Guitar,p4]
            P(env, 1, 4, 1, pianoShift) -- [1,0,4,Piano,p4]
        elseif playListSize(env) == 2 then
            -- 两个音：
            -- [1,0,4,Guitar,p4],[2,0,4,Guitar,p4]
            -- [1,0,4,Piano,p4],[2,0,4,Piano,p4]
            P(env, 1, 4, 1, pianoShift)
            P(env, 1, 4, 2, guitarShift)
            waitTime(env, 32)
            P(env, 2, 4, 1, pianoShift)
            P(env, 2, 4, 2, guitarShift)
        elseif playListSize(env) == 3 then
            -- 三个音：
            -- [1,0,4,Guitar,p4],[2,0,4,Guitar,p4,+12],[3,0,4,Guitar,p4]
            -- [1,0,4,Piano,p4],[2,2,2,Piano,p3,+12],[3,0,4,Piano,p4]
            P(env, 1, 4, 1, pianoShift)
            P(env, 3, 4, 1, pianoShift)
            P(env, 1, 4, 2, guitarShift)
            P(env, 2, 4, 2, guitarShift + 12)
            P(env, 3, 4, 2, guitarShift)
            waitTime(env, 32)
            P(env, 2, 3, 1, pianoShift + 12)
        elseif playListSize(env) == 4 then
            -- 四个音：
            -- [1,0,4,Guitar,p4],[2,0,4,Guitar,p4,+12],[3,0,4,Guitar,p4,+12],[4,0,4,Guitar,p4]
            -- [1,0,4,Piano,p4],[2,2,2,Piano,p3,+12],[3,2,2,Piano,p3,+12],[4,0,4,Piano,p4]
            P(env, 1, 4, 2, guitarShift)
            P(env, 2, 4, 2, guitarShift + 12)
            P(env, 3, 4, 2, guitarShift + 12)
            P(env, 4, 4, 2, guitarShift)
            P(env, 1, 4, 1, pianoShift)
            P(env, 4, 4, 1, pianoShift)
            waitTime(env, 32)
            P(env, 2, 3, 1, pianoShift + 12)
            P(env, 3, 3, 1, pianoShift + 12)
        elseif playListSize(env) == 5 then
            -- 五个音：
            -- [1,0,4,Guitar,p4],[2,0,4,Guitar,p4,+12],[3,0,4,Guitar,p4,+12],[4,0,4,Guitar,p4],[5,0,4,Guitar,p4]
            -- [1,0,4,Piano,p4],[2,2,2,Piano,p3,+12],[3,0,4,Piano,p3,+12],[4,0,4,Piano,p4],[5,0,4,Piano,p4]
            P(env, 1, 4, 2, guitarShift)
            P(env, 2, 4, 2, guitarShift + 12)
            P(env, 4, 4, 2, guitarShift)
            P(env, 5, 4, 2, guitarShift)
            waitTime(env, 32)
            P(env, 2, 3, 1, pianoShift + 12)
            P(env, 3, 4, 2, guitarShift + 12)
        end
    end
end
