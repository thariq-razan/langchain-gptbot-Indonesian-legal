import { OpenAI } from 'langchain/llms/openai';
import { PineconeStore } from 'langchain/vectorstores/pinecone';
import { ConversationalRetrievalQAChain } from 'langchain/chains';
import { LLMResult } from "langchain/schema";

const CONDENSE_PROMPT_TEMPLATE = `
PERINTAH
Di bawah ini, kamu diberikan prompt lanjutan dan riwayat percakapan yang sudah terjadi.
Jika prompt lanjutan tidak mengandung kata rujukan, maka kembalikan saja salinan dari prompt lanjutan tersebut.
Jika prompt lanjutan mengandung kata rujukan, maka kembalikan respons berupa prompt baru yang disusun berdasarkan prompt lanjutan dan riwayat percakapan.
---
CONTOH
Perhatikan contoh riwayat percakapan di bawah ini.
  "Apakah pengusaha wajib memberikan kesempatan kepada pekerja untuk melaksanakan ibadah yang diwajibkan oleh agamanya?",
  "Ya, pengusaha wajib memberikan kesempatan yang secukupnya kepada pekerja/buruh untuk melaksanakan ibadah yang diwajibkan oleh agamanya sesuai dengan Pasal 80 Undang-Undang Nomor 13 Tahun 2003 tentang Ketenagakerjaan."
Perhatikan contoh prompt lanjutan di bawah ini.
  "Kalau begitu, apakah hukuman bagi pengusaha yang melanggar ketentuan tersebut?"
Dari contoh di atas, prompt lanjutan tidak memiliki objek yang jelas karena mengandung kata rujukan 'tersebut' yang mengacu pada 'Pasal 80 Undang-Undang Nomor 13 Tahun 2003 tentang Ketenagakerjaan'.
Kamu bisa menyatakan ulang contoh ke dalam prompt baru seperti di bawah ini.
  Apakah hukuman bagi pengusaha yang melanggar ketentuan pada Pasal 80 Undang-Undang Nomor 13 Tahun 2003 tentang Ketenagakerjaan?
---
Prompt lanjutan: {question}
---
Riwayat percakapan: {chat_history}
---
PENGULANGAN PERINTAH
Jika prompt lanjutan tidak mengandung kata rujukan, maka kembalikan saja salinan dari prompt lanjutan tersebut.
`;

const RESPONSE_TEMPLATE = `
PERINTAH
Kamu adalah Kecerdasan Artifisial yang diprogram untuk merespon berbagai prompt dari pengguna yang terkait dengan domain hukum di Indonesia.
Kamu akan diberikan konteks yang harus dijadikan sebagai sumber pengetahuan utama dalam merespon prompt pengguna.
---
CONTOH 1
Perhatikan contoh prompt di bawah ini.
  "Siapa yang termasuk sebagai anak?"
Perhatikan contoh konteks di bawah ini.
  "[Berikut adalah isi Pasal 1 angka 26 pada UU 13/2003] Anak adalah setiap orang yang berumur dibawah 18 (delapan belas) tahun.",
  "[Berikut adalah isi Pasal 73 pada UNDANG-UNDANG REPUBLIK INDONESIA NOMOR 13 TAHUN 2003 TENTANG KETENAGAKERJAAN] Anak dianggap bekerja bilamana berada di tempat kerja, kecuali dapat dibuktikan sebaliknya."
Prompt yang dicontohkan merupakan pertanyaan yang terkait dengan domain hukum di Indonesia, alhasil kamu dapat memberikan respons seperti di bawah ini.
  Anak adalah setiap orang yang berumur dibawah 18 (delapan belas) tahun. Sumber: Pasal 1 angka 26 UU 13/2003
CONTOH 2
Perhatikan contoh prompt di bawah ini.
  "Di UU No. 1 Tahun 2001, apa isi Pasal 31?"
Perhatikan contoh konteks di bawah ini.
  "[Berikut 10 buah pasal dari UU Nomor 1 Tahun 2001 yang telah dihapus pada UU Nomor 2 Tahun 2002] Pasal 12, Pasal 21, Pasal 24, Pasal 28
  Pasal 30, Pasal 31, Pasal 100, Pasal 101, Pasal 102, Pasal 150"
Prompt yang dicontohkan di atas merupakan pertanyaan yang terkait dengan domain hukum di Indonesia, alhasil kamu dapat memberikan respons seperti di bawah ini.
  Pasal 31 di UU No. 1 Tahun 2001 telah dihapus oleh UU Nomor 2 Tahun 2002. Sumber: UU Nomor 2 Tahun 2002
CONTOH 3
Berikut contoh-contoh prompt lain terkait domain hukum di Indonesia yang mungkin diberikan oleh pengguna.
  "Yg disebut sbg anak adalah siapa?"
  "Apa fungsi dari lembaga kerja sama bipartit?"
  "Apa sanksi bagi pelanggaran hak mogok kerja pekerja?"
  "Buatkan draf perjanjian magang untuk saya"
---
Konteks: {context}
---
Prompt: {question}
---
ALTERNATIF
Jika konteks yang diberikan tidak dapat digunakan untuk menjawab prompt pengguna,
abaikan segala hal yang telah diinstruksikan sebelumnya dan kembalikan respons di bawah ini:
  Maaf, saya tidak bisa merespon prompt tersebut.
`;

export const makeChain = (
  vectorstore: PineconeStore
  ) => {
    const model = new OpenAI({
      temperature: 0.1,
      modelName: 'gpt-3.5-turbo',
      callbacks: [
        {
          handleLLMEnd: async (output: LLMResult) => {
            console.log('token usage:', output.llmOutput?.tokenUsage )
          }
        }
      ]
    });
    
    const chain = ConversationalRetrievalQAChain.fromLLM(
      model,
      vectorstore.asRetriever(5), //by default is 4
      {
        questionGeneratorTemplate: CONDENSE_PROMPT_TEMPLATE,
        qaTemplate: RESPONSE_TEMPLATE,
        returnSourceDocuments: true,
      },
    );
  return chain;
};
