"use client";

import { useState, useEffect } from "react";
import { X, Upload, FileText, CheckCircle, AlertCircle, Trash, Eye, EyeOff } from "lucide-react";

interface DocumentManagerProps {
  onClose: () => void;
}

interface UploadedDoc {
  filename: string;
  pages: number;
  chunks: number;
  uploaded_at: string;
}

export default function DocumentManager({ onClose }: DocumentManagerProps) {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [isUploading, setIsUploading] = useState(false);
  const [uploadStatus, setUploadStatus] = useState<'idle' | 'success' | 'error'>('idle');
  const [message, setMessage] = useState('');
  const [uploadedDocs, setUploadedDocs] = useState<UploadedDoc[]>([]);
  const [showDocuments, setShowDocuments] = useState(false);

  useEffect(() => {
    fetchDocuments();
  }, []);

  const fetchDocuments = async () => {
    try {
      const response = await fetch('http://localhost:8000/api/documents/list');
      const data = await response.json();
      setUploadedDocs(data.documents || []);
    } catch (error) {
      console.error('Failed to fetch documents:', error);
    }
  };

  const handleFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      const file = e.target.files[0];
      if (file.type === 'application/pdf') {
        setSelectedFile(file);
        setUploadStatus('idle');
        setMessage('');
      } else {
        setMessage('Please select a PDF file');
        setUploadStatus('error');
      }
    }
  };

  const handleUpload = async () => {
    if (!selectedFile) return;

    setIsUploading(true);
    setUploadStatus('idle');

    const formData = new FormData();
    formData.append('file', selectedFile);

    try {
      const response = await fetch('http://localhost:8000/api/documents/upload', {
        method: 'POST',
        body: formData,
      });

      const data = await response.json();

      if (response.ok) {
        setUploadStatus('success');
        setMessage(data.message);
        setSelectedFile(null);
        fetchDocuments();
        setTimeout(() => {
          setUploadStatus('idle');
          setMessage('');
        }, 3000);
      } else {
        setUploadStatus('error');
        setMessage(data.detail || 'Upload failed');
      }
    } catch (error) {
      setUploadStatus('error');
      setMessage('Network error. Please try again.');
    } finally {
      setIsUploading(false);
    }
  };

  const handleClearDocuments = async () => {
    if (!confirm('Are you sure you want to clear all documents?')) return;

    try {
      const response = await fetch('http://localhost:8000/api/documents/clear', {
        method: 'DELETE',
      });

      if (response.ok) {
        setMessage('All documents cleared successfully');
        setUploadStatus('success');
        fetchDocuments();
        setTimeout(() => {
          setUploadStatus('idle');
          setMessage('');
        }, 2000);
      }
    } catch (error) {
      setMessage('Failed to clear documents');
      setUploadStatus('error');
    }
  };

  return (
    <div className="fixed inset-0 bg-black/50 backdrop-blur-sm z-50 flex items-center justify-center p-4 animate-fadeIn">
      <div className="bg-card rounded-2xl shadow-2xl max-w-2xl w-full max-h-[90vh] overflow-hidden flex flex-col border border-border">
        {/* Header */}
        <div className="flex items-center justify-between p-6 border-b border-border">
          <div className="flex items-center gap-4">
            <div className="p-3 bg-primary rounded-xl shadow-sm">
              <FileText className="text-primary-foreground w-6 h-6" />
            </div>
            <div>
              <h2 className="text-2xl font-semibold text-foreground">Document Manager</h2>
              <p className="text-sm text-muted-foreground mt-0.5">Upload and manage PDFs for RAG</p>
            </div>
          </div>
          <button
            onClick={onClose}
            className="text-muted-foreground hover:text-foreground transition-colors rounded-lg p-2 hover:bg-accent"
            aria-label="Close"
          >
            <X className="w-6 h-6" />
          </button>
        </div>

        {/* Content */}
        <div className="flex-1 overflow-y-auto p-6 space-y-6">
          {/* Upload Area */}
          <div>
            <h3 className="text-lg font-semibold text-foreground mb-4">Upload Document</h3>
            <div className="border-2 border-dashed border-border hover:border-primary rounded-2xl p-10 text-center transition-all cursor-pointer bg-muted/20">
              <input
                type="file"
                accept=".pdf"
                onChange={handleFileSelect}
                className="hidden"
                id="file-upload"
              />
              <label
                htmlFor="file-upload"
                className="cursor-pointer flex flex-col items-center gap-4"
              >
                <div className="p-5 bg-muted rounded-2xl">
                  <Upload className="w-10 h-10 text-muted-foreground" />
                </div>
                <div>
                  <p className="font-semibold text-foreground text-lg mb-1">Click to select a PDF</p>
                  <p className="text-sm text-muted-foreground">or drag and drop here</p>
                </div>
              </label>
            </div>

            {selectedFile && (
              <div className="mt-4 flex items-center justify-between p-4 bg-muted/30 border border-border rounded-xl">
                <div className="flex items-center gap-3">
                  <FileText className="w-5 h-5 text-foreground" />
                  <div>
                    <p className="font-medium text-foreground">{selectedFile.name}</p>
                    <p className="text-sm text-muted-foreground">
                      {(selectedFile.size / 1024 / 1024).toFixed(2)} MB
                    </p>
                  </div>
                </div>
                <button
                  onClick={() => setSelectedFile(null)}
                  className="text-muted-foreground hover:text-foreground transition-colors p-1 hover:bg-accent rounded-lg"
                  aria-label="Remove file"
                >
                  <X className="w-5 h-5" />
                </button>
              </div>
            )}

            {message && (
              <div className={`mt-4 flex items-center gap-3 p-4 rounded-xl border ${
                uploadStatus === 'success' 
                  ? 'bg-green-50 border-green-200' 
                  : 'bg-red-50 border-red-200'
              }`}>
                {uploadStatus === 'success' ? (
                  <CheckCircle className="w-5 h-5 text-green-600 shrink-0" />
                ) : (
                  <AlertCircle className="w-5 h-5 text-red-600 shrink-0" />
                )}
                <p className={`text-sm leading-relaxed ${
                  uploadStatus === 'success' ? 'text-green-800' : 'text-red-800'
                }`}>
                  {message}
                </p>
              </div>
            )}

            <div className="mt-4">
              <button
                onClick={handleUpload}
                disabled={!selectedFile || isUploading}
                className="w-full bg-primary hover:bg-primary/90 disabled:bg-muted disabled:cursor-not-allowed text-primary-foreground font-medium py-3.5 px-4 rounded-xl transition-all shadow-md hover:shadow-lg active:scale-[0.98]"
              >
                {isUploading ? 'Uploading...' : 'Upload Document'}
              </button>
            </div>
          </div>

          {/* View Uploaded Documents */}
          <div>
            <button
              onClick={() => setShowDocuments(!showDocuments)}
              className="w-full flex items-center justify-between p-4 bg-muted/30 hover:bg-muted/50 rounded-xl transition-all border border-border"
            >
              <div className="flex items-center gap-3">
                <FileText className="w-5 h-5 text-foreground" />
                <span className="font-medium text-foreground">
                  Uploaded Documents ({uploadedDocs.length})
                </span>
              </div>
              {showDocuments ? (
                <EyeOff className="w-5 h-5 text-muted-foreground" />
              ) : (
                <Eye className="w-5 h-5 text-muted-foreground" />
              )}
            </button>

            {showDocuments && uploadedDocs.length > 0 && (
              <div className="mt-4 space-y-2 max-h-64 overflow-y-auto">
                {uploadedDocs.map((doc, index) => (
                  <div key={index} className="flex items-center gap-3 p-4 bg-card border border-border rounded-xl hover:border-primary transition-all">
                    <FileText className="w-5 h-5 text-foreground shrink-0" />
                    <div className="flex-1 min-w-0">
                      <p className="font-medium text-foreground truncate">{doc.filename}</p>
                      <p className="text-xs text-muted-foreground mt-0.5">
                        {doc.pages} pages • {doc.chunks} chunks
                      </p>
                    </div>
                    <CheckCircle className="w-5 h-5 text-green-600 shrink-0" />
                  </div>
                ))}
              </div>
            )}

            {showDocuments && uploadedDocs.length === 0 && (
              <div className="mt-4 text-center py-12 text-muted-foreground">
                <FileText className="w-12 h-12 mx-auto mb-3 opacity-50" />
                <p className="text-sm">No documents uploaded yet</p>
              </div>
            )}
          </div>

          {/* Info */}
          <div className="bg-blue-50 border border-blue-200 rounded-xl p-4">
            <p className="text-sm text-blue-800 leading-relaxed">
              <strong className="font-semibold">💡 Tip:</strong> After uploading, ask questions like "What does the document say about..." in your chat to use RAG.
            </p>
          </div>
        </div>

        {/* Footer */}
        <div className="p-6 border-t border-border bg-muted/20">
          <button
            onClick={handleClearDocuments}
            disabled={uploadedDocs.length === 0}
            className="w-full flex items-center justify-center gap-2 bg-destructive hover:bg-destructive/90 disabled:bg-muted disabled:cursor-not-allowed text-destructive-foreground font-medium py-3.5 px-4 rounded-xl transition-all shadow-md hover:shadow-lg active:scale-[0.98]"
          >
            <Trash className="w-5 h-5" />
            Clear All Documents
          </button>
        </div>
      </div>
    </div>
  );
}
