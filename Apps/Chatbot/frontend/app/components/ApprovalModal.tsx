"use client";

import { AlertTriangle, CheckCircle, XCircle } from "lucide-react";
import { PendingApproval } from "./ChatClient";

interface ApprovalModalProps {
  approval: PendingApproval;
  onApprove: () => void;
  onDecline: () => void;
}

export default function ApprovalModal({ approval, onApprove, onDecline }: ApprovalModalProps) {
  return (
    <div className="fixed inset-0 bg-black/50 backdrop-blur-sm z-50 flex items-center justify-center p-4 animate-fadeIn">
      <div className="bg-card rounded-2xl shadow-2xl max-w-md w-full p-6 border border-border animate-fadeIn">
        {/* Header */}
        <div className="flex items-start gap-4 mb-6">
          <div className="p-3 bg-yellow-100 rounded-xl shrink-0">
            <AlertTriangle className="text-yellow-600 w-6 h-6" />
          </div>
          <div className="flex-1">
            <h2 className="text-xl font-semibold text-foreground mb-1">Approval Required</h2>
            <p className="text-sm text-muted-foreground">Human-in-the-Loop Verification</p>
          </div>
        </div>

        {/* Content */}
        <div className="space-y-4 mb-6">
          <div className="bg-muted rounded-xl p-4 border border-border">
            <p className="text-xs font-medium text-muted-foreground uppercase tracking-wide mb-3">Action Details</p>
            <div className="space-y-3">
              <div className="flex justify-between items-center">
                <span className="text-sm text-muted-foreground">Type:</span>
                <span className="text-sm font-semibold text-foreground">Stock Purchase</span>
              </div>
              <div className="flex justify-between items-center">
                <span className="text-sm text-muted-foreground">Symbol:</span>
                <span className="text-base font-bold text-foreground">{approval.symbol}</span>
              </div>
              <div className="flex justify-between items-center">
                <span className="text-sm text-muted-foreground">Quantity:</span>
                <span className="text-sm font-semibold text-foreground">{approval.quantity} shares</span>
              </div>
            </div>
          </div>

          <div className="bg-yellow-50 border border-yellow-200 rounded-xl p-4">
            <p className="text-sm text-yellow-800 leading-relaxed">
              <strong className="font-semibold">⚠️ Warning:</strong> This action requires your explicit approval before proceeding.
            </p>
          </div>
        </div>

        {/* Actions */}
        <div className="flex gap-3">
          <button
            onClick={onDecline}
            className="flex-1 flex items-center justify-center gap-2 bg-muted hover:bg-muted/80 text-foreground font-medium py-3 px-4 rounded-xl transition-all border border-border active:scale-[0.98]"
          >
            <XCircle className="w-5 h-5" />
            Decline
          </button>
          <button
            onClick={onApprove}
            className="flex-1 flex items-center justify-center gap-2 bg-primary hover:bg-primary/90 text-primary-foreground font-medium py-3 px-4 rounded-xl transition-all shadow-md hover:shadow-lg active:scale-[0.98]"
          >
            <CheckCircle className="w-5 h-5" />
            Approve
          </button>
        </div>

        <p className="text-xs text-muted-foreground text-center mt-4">
          This is a simulated action for demonstration purposes
        </p>
      </div>
    </div>
  );
}
