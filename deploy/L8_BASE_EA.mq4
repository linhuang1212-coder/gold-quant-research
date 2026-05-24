//+------------------------------------------------------------------+
//| L8_BASE_EA.mq4                                                   |
//| L8_BASE + Cap80                                                   |
//| R41 K-Fold 6/6 PASS (CorrSh Mean=6.27, Min=1.14)                |
//| R144: +Rule B extreme protection, Sharpe +0.38                   |
//| R250 A1 (H1 EMA9>EMA21 过滤): code 已加, 但 R252 判定 isolated   |
//|   peak / 轻度过拟合, **默认关闭**, 仅作 ablation 用。            |
//| Chart: XAUUSD M15                                                |
//+------------------------------------------------------------------+
//| L8_BASE vs L7:                                                    |
//|   ADX: 18 -> 14, Normal Trail: 0.14/0.025, High: 0.06/0.008    |
//|   TATrail: OFF, MaxHold: 20                                       |
//| Rule B: R144 validated, skip 8 H1 bars after 3-sigma ATR spike   |
//| A1: R250 code 集成, R252 邻域扫描否决, 默认 false                |
//+------------------------------------------------------------------+
#property copyright "Gold Quant Research"
#property version   "3.21"
#property strict

//--- Trading params
extern double LotSize          = 0.01;
extern int    MagicNumber      = 20250427;
extern int    MaxSlippage       = 30;

//--- Keltner entry
extern int    KC_EMA_Period     = 25;
extern double KC_Multiplier     = 1.2;
extern int    ATR_Period        = 14;
extern int    ADX_Period        = 14;
extern double ADX_Threshold     = 14.0;
extern int    EMA100_Period     = 100;

//--- Exit params
extern double SL_ATR_Mult       = 3.5;
extern double TP_ATR_Mult       = 8.0;
extern int    MaxHold_Bars      = 20;
extern double Trail_Act_ATR     = 0.14;
extern double Trail_Dist_ATR    = 0.025;

//--- Regime Trail
extern double Regime_Low_Act    = 0.22;
extern double Regime_Low_Dist   = 0.04;
extern double Regime_Norm_Act   = 0.14;
extern double Regime_Norm_Dist  = 0.025;
extern double Regime_High_Act   = 0.06;
extern double Regime_High_Dist  = 0.008;

//--- H1 KC Filter
extern bool   H1_Filter_Enabled = true;
extern int    H1_KC_EMA_Period  = 15;
extern double H1_KC_Multiplier  = 2.0;
extern int    H1_ATR_Period     = 14;

//--- R250 A1: H1 EMA Trend Filter — DEFAULT DISABLED per R252 verdict.
//--- BUY 信号需 H1 EMA9 > EMA21 同向; SELL 信号需 H1 EMA9 < EMA21.
//--- R250 single-config 11y Sharpe gain looked promising, BUT R252 邻域扫描
//--- (EMA fast/slow ± 2 grid) 显示 (9,21) 是 isolated peak, 邻近 8/9 参数
//--- 都变差, 且 OOS 16 月增益接近 0 — 判定轻度过拟合, 暂不投产。
//--- 若要重新评估, 设 A1_TrendFilter_Enabled=true 单独跑 paper 至少 4 周。
extern bool   A1_TrendFilter_Enabled = false;
extern int    A1_EMA_Fast_Period     = 9;
extern int    A1_EMA_Slow_Period     = 21;

//--- EqCurve
extern bool   EqCurve_Enabled   = true;
extern int    EqCurve_LB        = 10;
extern double EqCurve_Cut       = 0.0;
extern double EqCurve_Red       = 0.0;

//--- Entry gap
extern double MinEntryGapHours  = 1.0;

//--- Choppy
extern double ChoppyThreshold   = 0.50;

//=== EXECUTION EDGE ===
extern bool   KCBW_Enabled      = false;
extern int    KCBW_Lookback     = 5;
extern double KCBW_Min_Ratio    = 0.0;

extern bool   MaxLoss_Enabled   = true;
extern double MaxLoss_USD       = 30.0;

extern bool   Session_Filter    = false;
extern int    Session_Skip_Start = 2;
extern int    Session_Skip_End   = 5;

//--- Rule B extreme protection (R144: Sharpe +0.38)
extern bool   RuleB_Enabled     = true;
extern double RuleB_ATR_Sigma   = 3.0;
extern int    RuleB_ATR_Lookback = 60;
extern int    RuleB_SkipBars    = 8;

//--- Global
datetime lastEntryTime  = 0;
datetime lastBarTime    = 0;
int      barsHeld       = 0;
double   entryATR       = 0;
double   trailStopPrice = 0;
double   extremePrice   = 0;
int      myTicket       = -1;

double   recentPnL[];
int      eqCurveCount   = 0;
bool     eqCurveSkip    = false;

double   atrHistory[];
int      atrHistCount   = 0;
int      ATR_HIST_SIZE  = 50;

int      ruleB_skipCount    = 0;
datetime ruleB_lastCheckBar = 0;

//+------------------------------------------------------------------+
bool IsExtremeMarket()
{
   if(!RuleB_Enabled) return false;
   double sum = 0, sum2 = 0;
   for(int i = 0; i < RuleB_ATR_Lookback; i++)
   {
      double v = iATR(Symbol(), PERIOD_H1, 14, i + 1);
      sum += v;
      sum2 += v * v;
   }
   double mean = sum / RuleB_ATR_Lookback;
   double var  = sum2 / RuleB_ATR_Lookback - mean * mean;
   double std  = MathSqrt(MathMax(var, 0.000001));
   double cur  = iATR(Symbol(), PERIOD_H1, 14, 0);
   return (cur > mean + RuleB_ATR_Sigma * std);
}

//+------------------------------------------------------------------+
int OnInit()
{
   ArrayResize(recentPnL, EqCurve_LB);
   ArrayInitialize(recentPnL, 0);
   ArrayResize(atrHistory, ATR_HIST_SIZE);
   ArrayInitialize(atrHistory, 0);

   Print("L8_BASE EA v3.10 | Lot=", LotSize,
         " ADX=", ADX_Threshold, " MH=", MaxHold_Bars,
         " Trail=", Trail_Act_ATR, "/", Trail_Dist_ATR,
         " KCBW=", KCBW_Enabled, " Cap=", MaxLoss_USD,
         " RuleB=", RuleB_Enabled, " Sigma=", RuleB_ATR_Sigma,
         " Skip=", RuleB_SkipBars);
   return INIT_SUCCEEDED;
}

//+------------------------------------------------------------------+
int GetATRRegime(double currentATR)
{
   if(atrHistCount < 10) return 1;
   double sorted[];
   int count = MathMin(atrHistCount, ATR_HIST_SIZE);
   ArrayResize(sorted, count);
   for(int i = 0; i < count; i++)
      sorted[i] = atrHistory[i % ATR_HIST_SIZE];
   ArraySort(sorted);

   int rank = 0;
   for(int i = 0; i < count; i++)
      if(sorted[i] <= currentATR) rank = i;

   double pct = (double)rank / (double)count;
   if(pct < 0.25) return 0;
   if(pct > 0.75) return 2;
   return 1;
}

//+------------------------------------------------------------------+
void GetRegimeTrailParams(int regime, double &act, double &dist)
{
   if(regime == 0)      { act = Regime_Low_Act;  dist = Regime_Low_Dist;  }
   else if(regime == 2) { act = Regime_High_Act; dist = Regime_High_Dist; }
   else                 { act = Regime_Norm_Act; dist = Regime_Norm_Dist; }
}

//+------------------------------------------------------------------+
string GetH1KCDirection()
{
   if(!H1_Filter_Enabled) return "PASS";

   double h1_close = iClose(Symbol(), PERIOD_H1, 1);
   double h1_ema   = iMA(Symbol(), PERIOD_H1, H1_KC_EMA_Period, 0, MODE_EMA, PRICE_CLOSE, 1);
   double h1_atr   = iATR(Symbol(), PERIOD_H1, H1_ATR_Period, 1);

   double kc_upper = h1_ema + H1_KC_Multiplier * h1_atr;
   double kc_lower = h1_ema - H1_KC_Multiplier * h1_atr;

   if(h1_close > kc_upper) return "BULL";
   if(h1_close < kc_lower) return "BEAR";
   return "NEUTRAL";
}

//+------------------------------------------------------------------+
//| R250 A1 趋势过滤: 检查 H1 EMA9 vs EMA21 是否同向                |
//| 返回 true 表示通过, false 表示应拦截入场                         |
//+------------------------------------------------------------------+
bool PassesA1TrendFilter(int signal)
{
   if(!A1_TrendFilter_Enabled) return true;

   double ema_fast = iMA(Symbol(), PERIOD_H1, A1_EMA_Fast_Period, 0,
                         MODE_EMA, PRICE_CLOSE, 1);
   double ema_slow = iMA(Symbol(), PERIOD_H1, A1_EMA_Slow_Period, 0,
                         MODE_EMA, PRICE_CLOSE, 1);

   if(signal == 1  && ema_fast <= ema_slow)
   {
      Print("A1 拦截 BUY: EMA9=", ema_fast, " <= EMA21=", ema_slow);
      return false;
   }
   if(signal == -1 && ema_fast >= ema_slow)
   {
      Print("A1 拦截 SELL: EMA9=", ema_fast, " >= EMA21=", ema_slow);
      return false;
   }
   return true;
}

//+------------------------------------------------------------------+
bool ShouldSkipByEqCurve()
{
   if(!EqCurve_Enabled) return false;
   if(eqCurveCount < EqCurve_LB) return false;

   double avg = 0;
   for(int i = 0; i < EqCurve_LB; i++)
      avg += recentPnL[i];
   avg /= EqCurve_LB;

   if(avg < EqCurve_Cut)
   {
      eqCurveSkip = true;
      return (EqCurve_Red == 0.0);
   }
   eqCurveSkip = false;
   return false;
}

//+------------------------------------------------------------------+
void RecordPnL(double pnl)
{
   for(int i = 0; i < EqCurve_LB - 1; i++)
      recentPnL[i] = recentPnL[i + 1];
   recentPnL[EqCurve_LB - 1] = pnl;
   eqCurveCount++;
}

//+------------------------------------------------------------------+
double CalcTrendScore()
{
   int up = 0, dn = 0;
   for(int i = 0; i < 20; i++)
   {
      double c0 = iClose(Symbol(), PERIOD_H1, i);
      double c1 = iClose(Symbol(), PERIOD_H1, i + 1);
      if(c0 > c1) up++;
      else dn++;
   }
   return (double)MathMax(up, dn) / 20.0;
}

//+------------------------------------------------------------------+
bool IsKCBandwidthExpanding()
{
   if(!KCBW_Enabled) return true;

   double bw[];
   ArrayResize(bw, KCBW_Lookback + 1);

   for(int i = 0; i <= KCBW_Lookback; i++)
   {
      double ema_val = iMA(Symbol(), PERIOD_H1, KC_EMA_Period, 0, MODE_EMA, PRICE_CLOSE, i + 1);
      double atr_val = iATR(Symbol(), PERIOD_H1, ATR_Period, i + 1);
      if(ema_val <= 0) return true;
      bw[i] = (KC_Multiplier * atr_val * 2.0) / ema_val;
   }

   double minBW = bw[1];
   for(int i = 2; i <= KCBW_Lookback; i++)
      if(bw[i] < minBW) minBW = bw[i];

   return (bw[0] > minBW);
}

//+------------------------------------------------------------------+
bool IsSessionFiltered()
{
   if(!Session_Filter) return false;
   int hour = TimeHour(TimeGMT());
   if(Session_Skip_Start <= Session_Skip_End)
      return (hour >= Session_Skip_Start && hour < Session_Skip_End);
   else
      return (hour >= Session_Skip_Start || hour < Session_Skip_End);
}

//+------------------------------------------------------------------+
int CheckKeltnerSignal()
{
   double ema_kc = iMA(Symbol(), PERIOD_H1, KC_EMA_Period, 0, MODE_EMA, PRICE_CLOSE, 0);
   double atr_h1 = iATR(Symbol(), PERIOD_H1, ATR_Period, 0);
   double adx    = iADX(Symbol(), PERIOD_H1, ADX_Period, PRICE_CLOSE, MODE_MAIN, 0);
   double ema100 = iMA(Symbol(), PERIOD_H1, EMA100_Period, 0, MODE_EMA, PRICE_CLOSE, 0);
   double close  = iClose(Symbol(), PERIOD_H1, 0);

   double kc_upper = ema_kc + KC_Multiplier * atr_h1;
   double kc_lower = ema_kc - KC_Multiplier * atr_h1;

   if(adx < ADX_Threshold) return 0;

   double trendScore = CalcTrendScore();
   if(trendScore < ChoppyThreshold) return 0;

   if(close > kc_upper && close > ema100) return 1;
   if(close < kc_lower && close < ema100) return -1;
   return 0;
}

//+------------------------------------------------------------------+
void CheckMaxLossCap()
{
   if(!MaxLoss_Enabled || MaxLoss_USD <= 0) return;
   if(myTicket < 0) return;
   if(OrderSelect(myTicket, SELECT_BY_TICKET, MODE_TRADES) == false) return;
   if(OrderCloseTime() > 0) return;

   double profit = OrderProfit() + OrderCommission() + OrderSwap();
   if(profit < -MaxLoss_USD)
   {
      double price;
      if(OrderType() == OP_BUY)
         price = MarketInfo(Symbol(), MODE_BID);
      else
         price = MarketInfo(Symbol(), MODE_ASK);

      bool closed = OrderClose(myTicket, OrderLots(), price, MaxSlippage, clrRed);
      if(closed)
      {
         Print("MaxLoss Cap! Loss=$", DoubleToStr(MathAbs(profit), 2),
               " > Cap=$", DoubleToStr(MaxLoss_USD, 2));
         RecordPnL(profit);
         barsHeld = 0;
         trailStopPrice = 0;
         extremePrice = 0;
         myTicket = -1;
      }
   }
}

//+------------------------------------------------------------------+
void ManagePosition()
{
   if(myTicket < 0) return;
   if(OrderSelect(myTicket, SELECT_BY_TICKET, MODE_TRADES) == false)
   {
      myTicket = -1;
      return;
   }
   if(OrderCloseTime() > 0)
   {
      double pnl = OrderProfit() + OrderCommission() + OrderSwap();
      RecordPnL(pnl);
      Print("Trade closed by SL/TP. PnL=$", DoubleToStr(pnl, 2));
      barsHeld = 0;
      trailStopPrice = 0;
      extremePrice = 0;
      myTicket = -1;
      return;
   }

   double posOpen = OrderOpenPrice();
   double posSL   = OrderStopLoss();
   double posTP   = OrderTakeProfit();
   int    posType = OrderType();

   double bid = MarketInfo(Symbol(), MODE_BID);
   double ask = MarketInfo(Symbol(), MODE_ASK);

   barsHeld++;

   if(posType == OP_BUY)
      extremePrice = MathMax(extremePrice, bid);
   else
      extremePrice = MathMin(extremePrice, ask);

   double currentATR = iATR(Symbol(), PERIOD_H1, ATR_Period, 0);
   if(currentATR <= 0) currentATR = entryATR;

   int regime = GetATRRegime(currentATR);
   double trail_act, trail_dist;
   GetRegimeTrailParams(regime, trail_act, trail_dist);

   double activateDist = trail_act * entryATR;
   double trailDist    = trail_dist * entryATR;

   if(posType == OP_BUY)
   {
      double floatProfit = bid - posOpen;
      if(floatProfit >= activateDist)
      {
         double newSL = extremePrice - trailDist;
         if(newSL > posSL && newSL < bid)
         {
            bool ok = OrderModify(myTicket, posOpen, newSL, posTP, 0, clrGreen);
            if(ok) trailStopPrice = newSL;
         }
      }
   }
   else
   {
      double floatProfit = posOpen - ask;
      if(floatProfit >= activateDist)
      {
         double newSL = extremePrice + trailDist;
         if(newSL < posSL && newSL > ask)
         {
            bool ok = OrderModify(myTicket, posOpen, newSL, posTP, 0, clrGreen);
            if(ok) trailStopPrice = newSL;
         }
      }
   }

   if(barsHeld >= MaxHold_Bars)
   {
      double price;
      if(posType == OP_BUY)
         price = MarketInfo(Symbol(), MODE_BID);
      else
         price = MarketInfo(Symbol(), MODE_ASK);

      Print("MaxHold reached (", MaxHold_Bars, " bars), closing");
      bool closed = OrderClose(myTicket, OrderLots(), price, MaxSlippage, clrYellow);
      if(closed)
      {
         double pnl = OrderProfit() + OrderCommission() + OrderSwap();
         RecordPnL(pnl);
         barsHeld = 0;
         trailStopPrice = 0;
         extremePrice = 0;
         myTicket = -1;
      }
   }
}

//+------------------------------------------------------------------+
bool HasOpenPosition()
{
   for(int i = OrdersTotal() - 1; i >= 0; i--)
   {
      if(OrderSelect(i, SELECT_BY_POS, MODE_TRADES))
      {
         if(OrderSymbol() == Symbol() && OrderMagicNumber() == MagicNumber)
         {
            myTicket = OrderTicket();
            return true;
         }
      }
   }
   return false;
}

//+------------------------------------------------------------------+
void OnTick()
{
   CheckMaxLossCap();

   datetime currentBar = iTime(Symbol(), PERIOD_M15, 0);
   if(currentBar == lastBarTime) return;
   lastBarTime = currentBar;

   // Rule B: check extreme market per H1 bar (not M15)
   datetime currentH1 = iTime(Symbol(), PERIOD_H1, 0);
   if(RuleB_Enabled && currentH1 != ruleB_lastCheckBar)
   {
      ruleB_lastCheckBar = currentH1;
      if(IsExtremeMarket())
      {
         ruleB_skipCount = RuleB_SkipBars;
         Print("Rule B: ATR spike detected, skipping ", RuleB_SkipBars, " H1 bars");
      }
      else if(ruleB_skipCount > 0)
      {
         ruleB_skipCount--;
      }
   }

   double h1_atr = iATR(Symbol(), PERIOD_H1, ATR_Period, 0);
   if(h1_atr > 0)
   {
      atrHistory[atrHistCount % ATR_HIST_SIZE] = h1_atr;
      atrHistCount++;
   }

   if(HasOpenPosition())
   {
      ManagePosition();
      return;
   }

   // Rule B: skip entry during cooldown
   if(ruleB_skipCount > 0) return;

   if(TimeCurrent() - lastEntryTime < (int)(MinEntryGapHours * 3600))
      return;

   if(IsSessionFiltered())
      return;

   int signal = CheckKeltnerSignal();
   if(signal == 0) return;

   string h1Dir = GetH1KCDirection();
   if(h1Dir != "PASS")
   {
      if(signal == 1 && h1Dir != "BULL") return;
      if(signal == -1 && h1Dir != "BEAR") return;
   }

   if(!PassesA1TrendFilter(signal))
      return;

   if(!IsKCBandwidthExpanding())
      return;

   if(ShouldSkipByEqCurve())
   {
      Print("EqCurve skip");
      return;
   }

   double atr_m15 = iATR(Symbol(), PERIOD_M15, ATR_Period, 0);
   if(atr_m15 < 0.1) return;

   entryATR = atr_m15;
   double sl_dist = atr_m15 * SL_ATR_Mult;
   double tp_dist = atr_m15 * TP_ATR_Mult;

   double lot = LotSize;
   if(EqCurve_Enabled && eqCurveSkip && EqCurve_Red > 0)
      lot = LotSize * EqCurve_Red;

   double ask_price = MarketInfo(Symbol(), MODE_ASK);
   double bid_price = MarketInfo(Symbol(), MODE_BID);

   int ticket = -1;
   if(signal == 1)
   {
      double sl = ask_price - sl_dist;
      double tp = ask_price + tp_dist;
      ticket = OrderSend(Symbol(), OP_BUY, lot, ask_price, MaxSlippage,
                         sl, tp, "L8v3 BUY", MagicNumber, 0, clrBlue);
      if(ticket > 0)
      {
         lastEntryTime = TimeCurrent();
         barsHeld = 0;
         extremePrice = bid_price;
         trailStopPrice = 0;
         myTicket = ticket;
         Print("BUY @ ", ask_price, " SL=", sl, " TP=", tp, " ATR=", atr_m15);
      }
   }
   else if(signal == -1)
   {
      double sl = bid_price + sl_dist;
      double tp = bid_price - tp_dist;
      ticket = OrderSend(Symbol(), OP_SELL, lot, bid_price, MaxSlippage,
                         sl, tp, "L8v3 SELL", MagicNumber, 0, clrRed);
      if(ticket > 0)
      {
         lastEntryTime = TimeCurrent();
         barsHeld = 0;
         extremePrice = ask_price;
         trailStopPrice = 0;
         myTicket = ticket;
         Print("SELL @ ", bid_price, " SL=", sl, " TP=", tp, " ATR=", atr_m15);
      }
   }
}
//+------------------------------------------------------------------+
