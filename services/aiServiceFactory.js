const config = require('../config/config');
const openaiService = require('./openaiService');
const ollamaService = require('./ollamaService');
const customService = require('./customService');
const azureService = require('./azureService');
const geminiService = require('./geminiService');
const claudeService = require('./claudeService');

class AIServiceFactory {
  static getService() {
    switch (config.aiProvider) {
      case 'ollama':
        return ollamaService;
      case 'openai':
      default:
        return openaiService;
      case 'custom':
        return customService;
      case 'azure':
        return azureService;
      case 'gemini':
        return geminiService;
      case 'claude':
        return claudeService;
    }
  }
}

module.exports = AIServiceFactory;
