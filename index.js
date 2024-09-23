const PORT = process.env.PORT || 5000;

const dotenv = require('dotenv');``
dotenv.config();

const express = require('express');
const app = express();
const http = require('http');
const server = http.createServer(app);

const cors = require('cors');
const path = require('path');
const xlsx = require('xlsx');

// MIDDLEWARES
app.use(cors());
app.use(express.json());

const io = require('socket.io')(server, { cors: { origin: "*" } });

// Path to your Excel file
const excelFilePath = path.join(__dirname, 'small data NYC EMS.xlsx');

// Function to load Excel data
let excelData = [];
try {
  const workbook = xlsx.readFile(excelFilePath);
  const sheetName = workbook.SheetNames[0]; // Assuming data is in the first sheet
  const worksheet = workbook.Sheets[sheetName];
  excelData = xlsx.utils.sheet_to_json(worksheet);  // Convert sheet to JSON
  console.log('Excel Data Loaded:', excelData.length, 'records');
} catch (error) {
  console.error('Error reading Excel file:', error);
}

// Routes    http://localhost:5000/
app.get('/', (req, res) => {           
  res.send('Hello, Excel Data is ready!');
});

// Example route to return first 10 records of Excel data    http://localhost:5000/api/data
app.get('/api/data', (req, res) => {
  if (excelData.length > 0) {
    res.json(excelData.slice(0, 10));  // Return first 10 records
  } else {
    res.status(500).send('Excel Data not loaded yet.');
  }
});

// Socket connection handling
server.listen(PORT, () => console.log(`socket server listening on port ${PORT}`));
let sid = '';

io.on('connection', (socket) => {
  console.log('Client connected');

  let uniqId = '';
  socket.on('send-id', (data) => {
    sid = data;
  });

  socket.on('send-coords', (data) => {
    const parsedData = JSON.parse(data);
    uniqId = parsedData.id;
    socket.to(sid).emit('reply', data);
  });

  socket.on('disconnect', () => {
    socket.to(sid).emit('disconnect-client', uniqId);
  });
});
