const Anthropic = require('@anthropic-ai/sdk');
const config = require('../config/config');
const paperlessService = require('./paperlessService');
const RestrictionPromptService = require('./restrictionPromptService');

class ClaudeService {
  constructor() {
    this.client = null;
  }

  initialize() {
    const apiKey = process.env.CLAUDE_API_KEY;
    if (!this.client && apiKey) {
      this.client = new Anthropic({ apiKey });
    }
  }

  async analyzeDocument(_content, existingTags = [], existingCorrespondentList = [], existingDocumentTypesList = [], id, customPrompt = null, options = {}) {
    try {
      this.initialize();
      const now = new Date();
      const timestamp = now.toLocaleString('de-DE', { dateStyle: 'short', timeStyle: 'short' });

      if (!this.client) {
        throw new Error('Claude client not initialized - missing API key');
      }

      console.log(`[DEBUG] [${timestamp}] Starting Claude analysis for document ${id}`);

      // 1. Download original PDF from Paperless
      const pdfBuffer = await paperlessService.downloadDocument(id);
      if (!pdfBuffer) {
        throw new Error(`Unable to download PDF for Document ${id}.`);
      }

      // 2. PDF als Base64 encodieren
      console.log(`[DEBUG] Converting PDF to Base64...`);
      const base64Pdf = pdfBuffer.toString('base64');

      // 3. Prompt zusammenbauen (identisch zu GeminiService)
      let customFieldsObj;
      try {
        customFieldsObj = JSON.parse(process.env.CUSTOM_FIELDS || '{"custom_fields": []}');
      } catch (error) {
        customFieldsObj = { custom_fields: [] };
      }

      const customFieldsTemplate = {};
      customFieldsObj.custom_fields.forEach((field, index) => {
        customFieldsTemplate[index] = {
          field_name: field.value,
          value: `Fill this field based on the document content. If the system prompt defines specific rules for '${field.value}', apply them EXACTLY. Otherwise, use the field name as guidance to extract the most relevant value from the document.`
        };
      });

      const customFieldsStr = '"custom_fields": ' + JSON.stringify(customFieldsTemplate, null, 2)
        .split('\n')
        .map(line => '    ' + line)
        .join('\n');

      let mustHavePrompt = config.mustHavePrompt
        .replace('%CUSTOMFIELDS%', customFieldsStr)
        .replace('%EXISTING_CORRESPONDENTS%', '');

      const activeSystemPrompt = (customPrompt || process.env.SYSTEM_PROMPT || '').replace(/\\n/g, '\n');
      let systemPrompt = '';

      if (config.useExistingData === 'yes') {
        const tagsList = existingTags.map(t => t.name || t).filter(Boolean).join(', ');
        const corrList = existingCorrespondentList.map(c => c.name || c).filter(Boolean).join(', ');
        const docTypesList = existingDocumentTypesList.map(d => d.name || d).filter(Boolean).join(', ');

        const tagsContext = config.restrictToExistingTags === 'yes'
          ? `Tags: ONLY use tags from this existing list: ${tagsList}. Do NOT return any tag not in this list. If no tag matches, return an empty array.`
          : `Pre-existing tags: ${tagsList}. Prefer these existing tags but you may also create new ones if they are a better fit.`;

        const corrContext = config.restrictToExistingCorrespondents === 'yes'
          ? `Correspondent: ONLY use correspondents from this existing list: ${corrList}. Do NOT create new correspondents. If no correspondent in the list matches the document's sender, return null for the correspondent field.`
          : `Pre-existing correspondents: ${corrList}. When identifying the correspondent, prefer an existing one if the sender is a close match. Use EXACTLY that name. Only return a new name if none of the existing correspondents are a reasonable match.`;

        const docTypeContext = config.restrictToExistingDocumentTypes === 'yes'
          ? `Document Type: ONLY use document types from this existing list: ${docTypesList}. Do NOT create new document types. If no document type matches, return null.`
          : `Pre-existing document types: ${docTypesList}. Prefer these existing document types but you may also create new ones if they are a better fit.`;

        systemPrompt = activeSystemPrompt.trimEnd() + '\n\n' + mustHavePrompt.trimEnd() + '\n\n' + `${tagsContext}\n\n${corrContext}\n\n${docTypeContext}`;
      } else {
        systemPrompt = activeSystemPrompt.trimEnd() + '\n\n' + mustHavePrompt;
      }

      systemPrompt = RestrictionPromptService.processRestrictionsInPrompt(
        systemPrompt,
        existingTags,
        existingCorrespondentList,
        config
      );

      if (options.extractContent) {
        systemPrompt += `\n\nIMPORTANT: This document has no pre-extracted text (OCR was empty). In addition to the metadata fields above, extract all readable text from the PDF and include it in the JSON response as an additional "extracted_content" field containing the full document text.`;
      }

      if (process.env.DEBUG_LOGGING === 'yes') {
        console.log(`[DEBUG] Full system prompt sent to Claude:\n${'─'.repeat(60)}\n${systemPrompt.replace(/\n\n/g, '\n \n')}\n${'─'.repeat(60)}`);
      }

      // 4. Claude API aufrufen
      const modelName = process.env.CLAUDE_MODEL || 'claude-sonnet-4-6';
      const thinkingEnabled = process.env.CLAUDE_EXTENDED_THINKING === 'yes';

      console.log(`[DEBUG] Requesting generation from ${modelName} (thinking: ${thinkingEnabled})...`);

      const thinkingBudget = parseInt(process.env.CLAUDE_THINKING_BUDGET || '10000');
      let maxTokens = parseInt(process.env.CLAUDE_MAX_TOKENS || '16000');
      if (thinkingEnabled && maxTokens <= thinkingBudget) {
        maxTokens = thinkingBudget + 1000;
        console.warn(`[WARN] CLAUDE_MAX_TOKENS (${maxTokens - 1000}) must be greater than CLAUDE_THINKING_BUDGET (${thinkingBudget}). Auto-corrected to ${maxTokens}.`);
      }

      const params = {
        model: modelName,
        max_tokens: maxTokens,
        system: systemPrompt,
        messages: [
          {
            role: 'user',
            content: [
              {
                type: 'document',
                source: {
                  type: 'base64',
                  media_type: 'application/pdf',
                  data: base64Pdf,
                },
              },
            ],
          },
        ],
      };

      if (thinkingEnabled) {
        params.thinking = {
          type: 'enabled',
          budget_tokens: thinkingBudget,
        };
        params.temperature = 1.0; // required by Anthropic when thinking is enabled
      } else if (process.env.CLAUDE_USE_TOP_P === 'yes') {
        // top_p and temperature are mutually exclusive in the Anthropic API
        params.top_p = parseFloat(process.env.CLAUDE_TOP_P || '0.9');
      } else {
        params.temperature = parseFloat(process.env.CLAUDE_TEMPERATURE || '1.0');
      }
      if (process.env.CLAUDE_TOP_K) {
        params.top_k = parseInt(process.env.CLAUDE_TOP_K);
      }

      const message = await this.client.messages.create(params);

      const metrics = {
        promptTokens: message.usage?.input_tokens || 0,
        completionTokens: message.usage?.output_tokens || 0,
        totalTokens: (message.usage?.input_tokens || 0) + (message.usage?.output_tokens || 0),
      };

      console.log(`[DEBUG] Token usage - Prompt: ${metrics.promptTokens}, Completion: ${metrics.completionTokens}, Total: ${metrics.totalTokens}`);

      // 5. Text-Block aus Response extrahieren (bei Thinking gibt es mehrere Content-Blöcke)
      const textBlock = message.content.find(b => b.type === 'text');
      if (!textBlock) {
        throw new Error('No text block in Claude response');
      }
      const text = textBlock.text;

      if (process.env.DEBUG_LOGGING === 'yes') {
        console.log(`[DEBUG] Raw Claude response:\n${'─'.repeat(60)}\n${text}\n${'─'.repeat(60)}`);
      }

      // 6. JSON bereinigen und parsen
      let jsonContent = text.replace(/```json\n?/gi, '').replace(/```\n?/g, '').trim();

      let parsedResponse;
      try {
        parsedResponse = JSON.parse(jsonContent);
      } catch (error) {
        console.error('Failed to parse JSON response:', error);
        throw new Error('Invalid JSON response from Claude');
      }

      if (parsedResponse.reasoning) {
        console.log(`[DEBUG] AI reasoning: ${parsedResponse.reasoning}`);
        delete parsedResponse.reasoning;
      }

      return {
        document: parsedResponse,
        metrics: metrics,
        truncated: false,
      };

    } catch (error) {
      console.error('Failed to analyze document with Claude:', error);
      return {
        document: { tags: [], correspondent: null },
        metrics: { promptTokens: 0, completionTokens: 0, totalTokens: 0 },
        error: error.message,
      };
    }
  }

  async checkStatus() {
    try {
      this.initialize();
      if (!this.client) throw new Error('Client not initialized');
      await this.client.messages.create({
        model: process.env.CLAUDE_MODEL || 'claude-sonnet-4-6',
        max_tokens: 10,
        messages: [{ role: 'user', content: 'Test' }],
      });
      return { status: 'ok', model: process.env.CLAUDE_MODEL || 'claude-sonnet-4-6' };
    } catch (error) {
      return { status: 'error', error: error.message };
    }
  }
}

module.exports = new ClaudeService();
