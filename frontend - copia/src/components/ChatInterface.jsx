import React, { useState, useRef, useEffect } from 'react';
import { Box, VStack, Input, Button, Text, useColorModeValue, Flex, Avatar } from "@chakra-ui/react";
import { Send, Bot } from 'lucide-react';
import axios from 'axios';

const ChatInterface = () => {
  const [query, setQuery] = useState('');
  const [messages, setMessages] = useState([]);
  const [isLoading, setIsLoading] = useState(false);
  const messagesEndRef = useRef(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!query.trim()) return;

    const userMessage = query.trim();
    setQuery('');
    setMessages(prev => [...prev, { type: 'user', content: userMessage }]);
    setIsLoading(true);

    try {
      const result = await axios.post(
        `${import.meta.env.VITE_API_URL}/query`,
        { question: userMessage },
        {
          headers: { 'Content-Type': 'application/json' },
          timeout: 600000
        }
      );

      if (result.data && result.data.response) {
        setMessages(prev => [...prev, { type: 'bot', content: result.data.response }]);
      }
    } catch (error) {
      console.error('Error:', error);
      const errorMessage = 'Lo siento, hubo un error al procesar tu consulta. Por favor, intenta de nuevo.';
      setMessages(prev => [...prev, { type: 'bot', content: errorMessage }]);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <Box
      maxW="1200px"
      mx="auto"
      bg={useColorModeValue('gray.100', 'gray.800')}
      borderRadius="xl"
      shadow="xl"
      overflow="hidden"
      h="400px"
      display="flex"
      flexDirection="column"
      border="1px solid"
      borderColor={useColorModeValue('gray.300', 'gray.700')}
    >
      <Box
        flex="1"
        overflowY="auto"
        p={4}
        bg={useColorModeValue('white', 'gray.900')}
      >
        <VStack spacing={4} align="stretch">
          {/* Mensaje de bienvenida */}
          <Flex gap={2}>
            <Avatar
              icon={<Bot size={20} />}
              bg="blue.500"
              color="white"
              size="sm"
            />
            <Box
              bg={useColorModeValue('blue.50', 'blue.900')}
              p={3}
              borderRadius="lg"
              maxW="80%"
              fontSize="sm"
            >
              <Text>
                ¡Bienvenido a CineExplorer! Puedo responder preguntas sobre las películas mostradas arriba.
                Por ejemplo, prueba preguntando "¿De qué trata la película Dune?" o "¿Cuáles son los personajes principales de Rush Hour?"
              </Text>
            </Box>
          </Flex>

          {/* Mensajes */}
          {messages.map((message, index) => (
            <Flex
              key={index}
              gap={2}
              justify={message.type === 'user' ? 'flex-end' : 'flex-start'}
            >
              {message.type === 'bot' && (
                <Avatar
                  icon={<Bot size={20} />}
                  bg="blue.500"
                  color="white"
                  size="sm"
                />
              )}
              <Box
                bg={message.type === 'user'
                  ? useColorModeValue('blue.500', 'blue.600')
                  : useColorModeValue('blue.50', 'blue.900')
                }
                color={message.type === 'user' ? 'white' : 'inherit'}
                p={3}
                borderRadius="lg"
                maxW="80%"
                fontSize="sm"
              >
                <Text>{message.content}</Text>
              </Box>
              {message.type === 'user' && (
                <Avatar
                  bg="gray.500"
                  size="sm"
                />
              )}
            </Flex>
          ))}

          {isLoading && (
            <Flex gap={2}>
              <Avatar
                icon={<Bot size={20} />}
                bg="blue.500"
                color="white"
                size="sm"
              />
              <Box
                bg={useColorModeValue('blue.50', 'blue.900')}
                p={3}
                borderRadius="lg"
                maxW="80%"
                fontSize="sm"
              >
                <Text>Procesando tu consulta. Esto puede tomar 2 o más minutos...</Text>
              </Box>
            </Flex>
          )}
          <div ref={messagesEndRef} />
        </VStack>
      </Box>

      {/* Input área */}
      <Box p={3} borderTop="1px" borderColor={useColorModeValue('gray.300', 'gray.700')} bg={useColorModeValue('gray.200', 'gray.800')}>
        <form onSubmit={handleSubmit}>
          <Flex gap={2}>
            <Input
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              placeholder="Haz una pregunta sobre una película..."
              size="md"
              disabled={isLoading}
              bg={useColorModeValue('gray.50', 'gray.900')}
            />
            <Button
              type="submit"
              colorScheme="blue"
              size="md"
              isLoading={isLoading}
              leftIcon={<Send size={16} />}
            >
              Enviar
            </Button>
          </Flex>
        </form>
      </Box>
    </Box>
  );
};

export default ChatInterface;