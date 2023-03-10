export default {
  name: 'game',
  type: 'document',
  title: 'Game',
  fields: [
    {
      title: 'History',
      name: 'history',
      type: 'array',
      of: [
        {
          type: 'object',
          fields: [
            {
              name: 'row',
              type: 'number',
            },
            {
              name: 'col',
              type: 'number',
            },
          ],
        },
      ],
    },
  ],
}
