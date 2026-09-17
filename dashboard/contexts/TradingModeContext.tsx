"use client";

import { createContext, useContext, useState, useCallback, type ReactNode } from "react";

export type TradingMode = "test" | "live";

interface TradingModeContextValue {
  mode: TradingMode;
  setMode: (mode: TradingMode) => void;
  dialogError: string | null;
  setDialogError: (msg: string | null) => void;
  showDialog: boolean;
  setShowDialog: (v: boolean) => void;
}

const TradingModeContext = createContext<TradingModeContextValue | null>(null);

export function TradingModeProvider({ children }: { children: ReactNode }) {
  const [mode, setModeInternal] = useState<TradingMode>("test");
  const [dialogError, setDialogError] = useState<string | null>(null);
  const [showDialog, setShowDialog] = useState(false);

  const setMode = useCallback((newMode: TradingMode) => {
    if (newMode === "live") {
      // Check for Zerodha keys - only needed for REAL money trading
      const apiKey = process.env.NEXT_PUBLIC_ZERODHA_API_KEY;
      const apiSecret = process.env.NEXT_PUBLIC_ZERODHA_API_SECRET;
      if (!apiKey || !apiSecret) {
        setDialogError(
          "LIVE MODE REQUIRES ZERODHA API KEYS\n\n" +
          "This mode executes REAL trades with real money via Zerodha.\n\n" +
          "For paper trading with live data, use PAPER mode (yellow button).\n\n" +
          "To enable live trading, set NEXT_PUBLIC_ZERODHA_API_KEY and " +
          "NEXT_PUBLIC_ZERODHA_API_SECRET in your .env.local file."
        );
        setShowDialog(true);
        return;
      }
    }
    setModeInternal(newMode);
  }, []);

  return (
    <TradingModeContext.Provider value={{ mode, setMode, dialogError, setDialogError, showDialog, setShowDialog }}>
      {children}
    </TradingModeContext.Provider>
  );
}

export function useTradingMode() {
  const ctx = useContext(TradingModeContext);
  if (!ctx) throw new Error("useTradingMode must be used within TradingModeProvider");
  return ctx;
}
