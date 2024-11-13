// app/api/start-server/route.ts
import { NextResponse } from 'next/server';
import { exec } from 'child_process';

export async function POST() {
  return new Promise((resolve) => {
    exec('python -m uvicorn app:app --host 127.0.0.1 --port 8000 --reload', (error, stdout, stderr) => {
      if (error) {
        console.error(`Error starting FastAPI server: ${stderr}`);
        resolve(NextResponse.json({ message: `Failed to start FastAPI server: ${stderr}` }, { status: 500 }));
      } else {
        console.log(`FastAPI server started: ${stdout}`);
        resolve(NextResponse.json({ message: 'FastAPI server started successfully' }));
      }
    });
  });
}

